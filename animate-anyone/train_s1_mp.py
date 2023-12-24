import sys
import time
from os.path import join

import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from accelerate import Accelerator
from einops import rearrange, repeat
import torch.nn.functional as F
from tqdm import tqdm

from datasets.jafarin import JafarinVideoDataset
from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet
from util import (get_models, retrieve_train_timesteps, get_initial_noise_latents, get_conditioning_embeddings,
                   encode_images, infer_fixed_sample_mp, retrieve_inference_timesteps, load_mm)

torch.manual_seed(17)

# model parallel implementation for stage one trainer

# save_model_checkpoint saves the model as a checkpoint
def save_model_checkpoint(ckpt_dir, stage, pose_guider_net, reference_net, video_net, optimizer):
    states = {
        'pose_guider_net': pose_guider_net.state_dict(),
        'reference_net': reference_net.state_dict(),
        'video_net': video_net.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(states, join(ckpt_dir, f'ckpt_s{stage}_t{int(time.time())}.pt'))

# gets the dataloader for the image dataset
def get_image_dataloader(batch_size, num_frames):
    dataset = JafarinVideoDataset('../datasets/TikTok_dataset/TikTok_dataset', num_frames)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [.95, .05])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    return train_dataloader, val_dataloader

# get_validation_loss gets the loss of the models on the validation set
def get_validation_loss(val_dataloader, num_frames, pose_guider_net: PoseGuider, reference_net: ReferenceNet, video_net: VideoNet, vision_encoder, vae, vae_scaling_factor, skip_temporal_attn=True):
    total_loss, total_items = 0, 0
    pose_guider_net.eval()
    reference_net.eval()
    video_net.eval()
    for _, data in enumerate(val_dataloader):
        if data[0] is None:
            continue

        pose_data, raw_img_data, ref_img_data = data[0].to('cuda:0'), data[1].to('cuda:0'), data[2].to('cuda:0')

        # we stack the pose and images to batch it through poseguider
        pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
        img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

        # get conditioning embeddings
        clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img_data, num_frames, vae_scaling_factor, 'cuda:0')

        # forward pass with no gradients and calculate loss
        with torch.no_grad():
            # 3) generate pose guided images
            pose_embeddings = pose_guider_net(pose_stack_data).to('cuda:1')

            # 4) generate embeddings from reference net
            reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
            
            # reference_frame_embeddings is the reference embeddings repeated for each frame
            reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames).to('cuda:1') for ref_emb in reference_embeddings]

            # 5) generate the noisy latents
            initial_noise_shape = (img_stack_data.shape[0], num_channels_latent, latent_height, latent_width)
            initial_noise_latents = get_initial_noise_latents(initial_noise_shape, scheduler, 'cuda:1')

            # 5.1) generate train timesteps
            timesteps = retrieve_inference_timesteps(scheduler, img_stack_data.shape[0] - 1, 'cuda:1') # -1 is because scheulder adds additional step

            # 5.2) generate image latents
            img_latents = encode_images(vae, img_stack_data, vae_scaling_factor).to('cuda:1')

            # 5.3) generate conditioned noise
            conditioned_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, timesteps)
            conditioned_noise_latents = conditioned_noise_latents + pose_embeddings

            # 6) predict noise for video frames
            clip_raw_frame_embeddings = clip_raw_frame_embeddings.to('cuda:1')
            noise_pred = video_net(conditioned_noise_latents, timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn)
            loss = F.mse_loss(noise_pred, initial_noise_latents)
            loss_item = loss.detach().item()
            total_loss += loss_item
            total_items += 1

    # after finding the average of entire validation set
    val_loss = total_loss / total_items
    accelerator.log({"stage_one_val_loss": val_loss})
    tqdm.write(f'val loss: {val_loss}')

if __name__ == '__main__':
    ckpt_dir = '../ckpts'
    # start_ckpt = '../ckpts/ckpt_s1_t1703121393_v2.pt'
    start_ckpt = ''
    stage_one_batch_size = 8
    stage_one_steps = 200000 # we manually tune steps to the equivalent of steps*batch=30000*64 (orig 16k, moved to 20k)
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 2
    inference_steps = 50
    learning_rate = 1e-5

    # configuration_values
    config = {
        "gradient_accumulation_steps": 2,
        "mixed_precision": 'bf16'
    }

    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        log_with="wandb",
    )

    # ensure that for model parallel at least two GPUs are available
    if torch.cuda.device_count() < 2:
        print('[!] model parallel trainer requires at least two GPUs')
        sys.exit()

    # init wandb tracking
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='animate',
            config={'num_gpus': accelerator.num_processes, 'num_frames': num_frames, 'learning_rate': learning_rate,
                        'batch_size': stage_one_batch_size, 'steps': stage_one_steps, 'mode': 'model_parallel', 'stage': 1,
                        'start_ckpt': start_ckpt, 'vnet_version': 4}
        )

    # get the models
    models = get_models(num_channels_latent, num_frames, 'cpu', ckpt=start_ckpt)
    
    # get relevant models and settings from the models, all models on first gpu except video net
    vision_processor: CLIPImageProcessor = models['vision_processor']
    vision_encoder: CLIPVisionModel = models['vision_encoder'].to('cuda:0', dtype=torch.bfloat16)
    vae: AutoencoderKL = models['vae'].to('cuda:0', dtype=torch.bfloat16)
    pose_guider_net: PoseGuider = models['pose_guider_net'].to('cuda:0')
    reference_net: ReferenceNet = models['reference_net'].to('cuda:0')
    video_net: VideoNet = models['video_net'].to('cuda:1')
    scheduler: PNDMScheduler = models['scheduler']
    vae_scaling_factor = vae.config.scaling_factor
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scaling_factor)

    scheduler.set_timesteps(inference_steps, device='cuda:1')
    video_net.batch_size = stage_one_batch_size

    # define dataloader and optimizer for stage one training
    train_dataloader, val_dataloader = get_image_dataloader(stage_one_batch_size, num_frames)

    pose_guider_net, reference_net, video_net, train_dataloader = accelerator.prepare(
        pose_guider_net, reference_net, video_net, train_dataloader,
        device_placement=[False, False, False, False]
    )
    optimizer = torch.optim.AdamW(
        list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
        lr=learning_rate,
    )
    optimizer = accelerator.prepare(optimizer)

    '''
    Begin stage one of training
    - stage one training objective is generating good images from poses (no temporal consistency)
    '''

    stage_one_train, global_step = True, 0
    with tqdm(total=stage_one_steps) as pbar:
        while stage_one_train:
            # train until global step is 30,000
            pose_guider_net.train()
            reference_net.train()
            video_net.train()
            for step, data in enumerate(train_dataloader):
                pose_data, raw_img_data, ref_img_data = data[0].to('cuda:0'), data[1].to('cuda:0'), data[2].to('cuda:0')

                # we stack the pose and images to batch it through poseguider
                pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
                img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

                # get conditioning embeddings
                clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img_data, num_frames, vae_scaling_factor, 'cuda:0')
                
                # generate noise, timesteps, image latents, and img with noise
                initial_noise_shape = (img_stack_data.shape[0], num_channels_latent, latent_height, latent_width)
                initial_noise_latents = get_initial_noise_latents(initial_noise_shape, scheduler, 'cuda:1')
                train_timesteps = retrieve_train_timesteps(scheduler, img_stack_data.shape[0], 'cuda:1')
                img_latents = encode_images(vae, img_stack_data, vae_scaling_factor).to('cuda:1')
                img_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)

                # 3) generate pose guided images
                pose_embeddings = pose_guider_net(pose_stack_data).to('cuda:1')

                # 4) generate embeddings from reference net
                reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
                
                # reference_frame_embeddings is the reference embeddings repeated for each frame
                reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames).to('cuda:1') for ref_emb in reference_embeddings]

                # 6) predict noise for video frames
                clip_raw_frame_embeddings = clip_raw_frame_embeddings.to('cuda:1')
                conditioned_noise_latents = img_noise_latents + pose_embeddings
                noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
                loss = F.mse_loss(noise_pred, initial_noise_latents)
                loss.backward()

                # backward optimizer pass
                torch.nn.utils.clip_grad_norm_(list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
                                            1.0)
                optimizer.step()
                optimizer.zero_grad()

                # log loss + update global step
                loss_item = loss.detach().item()
                accelerator.log({"stage_one_train_loss": loss_item})
                
                if global_step % 3000 == 0:
                    out_path = join('../out/train', f'train_mp_{global_step}.png')
                    tqdm.write('generating evaluation image...')
                    initial_noise_shape = (1, num_channels_latent, latent_height, latent_width)
                    infer_fixed_sample_mp(initial_noise_shape, 1, inference_steps, scheduler, vision_processor,
                            vision_encoder, vae, vae_scaling_factor, pose_guider_net, reference_net,
                            video_net, vae_image_processor, out_path)
                    tqdm.write('completed evaluation image...')

                tqdm.write(f'step: {global_step} loss: {loss_item}')
                global_step += stage_one_batch_size
                pbar.update(stage_one_batch_size)

                if global_step >= stage_one_steps:
                    stage_one_train = False
                    break

            # evaluate loss on the validation set
            get_validation_loss(val_dataloader, num_frames, pose_guider_net, reference_net, video_net, vision_encoder, vae, vae_scaling_factor)


    # checkpoint models after stage one
    save_model_checkpoint(ckpt_dir, 1, pose_guider_net, reference_net, video_net, optimizer)
