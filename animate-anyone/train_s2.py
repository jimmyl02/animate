import time
from os.path import join

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from tqdm import tqdm

from datasets.jafarin import JafarinVideoDataset
from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet
from util import (get_models, retrieve_train_timesteps, get_initial_noise_latents, get_conditioning_embeddings, encode_images, 
                  infer_fixed_sample, load_mm)

torch.manual_seed(17)

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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

# get_validation_loss gets the loss of the models on the validation set
def get_validation_loss(val_dataloader, num_frames, pose_guider_net: PoseGuider, reference_net: ReferenceNet, 
                        video_net: VideoNet, vision_encoder, vae, vae_scaling_factor, device, skip_temporal_attn=True):
    total_loss, total_items = 0, 0
    pose_guider_net.eval()
    reference_net.eval()
    video_net.eval()
    for step, data in enumerate(val_dataloader):
        if data[0] is None:
            continue

        pose_data, raw_img_data, ref_img_data = data[0].to(device), data[1].to(device), data[2].to(device)

        # we stack the pose and images to batch it through poseguider
        pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
        img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

        # get conditioning embeddings
        clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img_data, num_frames, vae_scaling_factor, device)

        # forward pass with no gradients and calculate loss
        with torch.no_grad():
            # 3) generate pose guided images
            pose_embeddings = pose_guider_net(pose_stack_data)

            # 4) generate embeddings from reference net
            reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
            
            # reference_frame_embeddings is the reference embeddings repeated for each frame
            reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

            # 5) generate the noisy latents
            initial_noise_shape = (img_stack_data.shape[0], num_channels_latent, latent_height, latent_width)
            initial_noise_latents = get_initial_noise_latents(initial_noise_shape, scheduler, device)

            # 5.1) generate train timesteps
            train_timesteps = retrieve_train_timesteps(scheduler, img_stack_data.shape[0], device)

            # 5.2) generate image latents
            img_latents = encode_images(vae, img_stack_data, vae_scaling_factor)

            # 5.3) generate conditioned noise
            conditioned_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)
            conditioned_noise_latents = conditioned_noise_latents + pose_embeddings

            # 6) predict noise for video frames
            noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn)
            loss = F.mse_loss(noise_pred, initial_noise_latents)
            loss_item = loss.detach().item()
            total_loss += loss_item
            total_items += 1

    # after finding the average of entire validation set
    val_loss = total_loss / total_items
    accelerator.log({"stage_two_val_loss": val_loss})
    tqdm.write(f'val loss: {val_loss}')

if __name__ == '__main__':
    ckpt_dir = '../ckpts'
    stage_two_batch_size = 1
    stage_two_steps = 40000 # TODO: adjust step count
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 8
    inference_steps = 50
    learning_rate = 1e-5

    # configuration_values
    config = {
        "gradient_accumulation_steps": 2,
        "mixed_precision": 'bf16'
    }

    # define the accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        log_with="wandb",
    )
    device = accelerator.device

    # init wandb tracking
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='animate_s2',
            config={'num_gpus': accelerator.num_processes, 'num_frames': num_frames, 'learning_rate': learning_rate,
                        'batch_size': stage_two_batch_size, 'steps': stage_two_steps, 'mode': 'dataparallel_gpu', 'stage': 2}
        )

    # get the models
    models = get_models(num_channels_latent, num_frames, device, ckpt='../ckpts/ckpt_s1_t1703297604_v3.pt')
    
    # get relevant models and settings from the models
    vision_processor: CLIPImageProcessor = models['vision_processor']
    vision_encoder: CLIPVisionModel = models['vision_encoder']
    vae: AutoencoderKL = models['vae']
    pose_guider_net: PoseGuider = models['pose_guider_net']
    reference_net: ReferenceNet = models['reference_net']
    video_net: VideoNet = models['video_net']
    scheduler: PNDMScheduler = models['scheduler']
    vae_scaling_factor = vae.config.scaling_factor
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scaling_factor)

    # load mm pretrained weights from animatediff
    load_mm(video_net, torch.load('../ckpts/v3_sd15_mm.ckpt'))

    scheduler.set_timesteps(inference_steps, device=device)

    # enable memory efficient attention (requires xformers)
    reference_net.unet.enable_xformers_memory_efficient_attention()
    video_net.unet.enable_xformers_memory_efficient_attention()

    '''
    Begin stage two of training
    - stage one training objective is generating good images from poses (no temporal consistency)
    '''

    # disable gradients for reference net, pose guider, and video net
    reference_net.requires_grad_(False)
    pose_guider_net.requires_grad_(False)
    video_net.requires_grad_(False)

    # re-enable weights only for temporal attention modules
    for i in range(len(video_net.ref_cond_attn_blocks)):
        # update the number of frames
        video_net.ref_cond_attn_blocks[i].tam.requires_grad_(True)

    # prepare models, dataloader, and optimizer for training
    train_dataloader, val_dataloader = get_image_dataloader(stage_two_batch_size, num_frames)
    pose_guider_net, reference_net, video_net, train_dataloader = accelerator.prepare(
        pose_guider_net, reference_net, video_net, train_dataloader
    )
    optimizer = torch.optim.AdamW(
        list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
        lr=learning_rate,
    )
    optimizer = accelerator.prepare(optimizer)

    # move vae and text encoder to correct device and dtype
    vision_encoder.to(device, dtype=torch.bfloat16)
    vae.to(device, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        pbar = tqdm(total=stage_two_steps)

    stage_two_train, global_step = True, 0
    while stage_two_train:
        # train until global step is 30,000
        reference_net.eval()
        pose_guider_net.eval()
        video_net.train()
        video_net.batch_size = stage_two_batch_size
        for step, data in enumerate(train_dataloader):
            pose_data, raw_img_data, ref_img_data = data[0].to(device), data[1].to(device), data[2].to(device)

            # we stack the pose and images to batch it through poseguider
            pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
            img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

            # get conditioning embeddings
            clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img_data, num_frames, vae_scaling_factor, device)
            
            # generate noise, timesteps, image latents, and img with noise
            initial_noise_shape = (stage_two_batch_size * num_frames, num_channels_latent, latent_height, latent_width)
            initial_noise_latents = get_initial_noise_latents(initial_noise_shape, scheduler, device)
            train_timesteps = retrieve_train_timesteps(scheduler, stage_two_batch_size * num_frames, device)
            img_latents = encode_images(vae, img_stack_data, vae_scaling_factor)
            img_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)

            with torch.no_grad():
                # 3) generate pose guided images
                pose_embeddings = pose_guider_net(pose_stack_data)

                # 4) generate embeddings from reference net
                reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
                
                # reference_frame_embeddings is the reference embeddings repeated for each frame
                reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

            with accelerator.accumulate(video_net):
                # 6) predict noise for video frames
                conditioned_noise_latents = img_noise_latents + pose_embeddings
                noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=False)
                loss = F.mse_loss(noise_pred, initial_noise_latents)
                avg_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
                accelerator.backward(loss)

                # backward optimizer pass
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
                                            1.0)
                optimizer.step()
                optimizer.zero_grad()

            # log loss + update global step
            accelerator.log({"stage_two_train_loss": avg_loss})
            
            if global_step % 1000 == 0 and accelerator.is_main_process:
                out_path = join('../out/train', f'train_p{accelerator.num_processes}_{global_step}.png')
                tqdm.write('generating evaluation image...')
                infer_noise_shape = (num_frames, num_channels_latent, latent_height, latent_width)
                infer_fixed_sample(infer_noise_shape, num_frames, inference_steps, scheduler, vision_processor,
                        vision_encoder, vae, vae_scaling_factor, pose_guider_net, reference_net,
                        video_net, vae_image_processor, out_path, device)
                tqdm.write('completed evaluation image...')

            global_step += accelerator.num_processes * stage_two_batch_size
            if accelerator.is_main_process:
                tqdm.write(f'step: {global_step} loss: {avg_loss}')
                pbar.update(accelerator.num_processes * stage_two_batch_size)

            if global_step >= stage_two_steps:
                stage_two_train = False
                break

        # evaluate loss on the validation set
        if accelerator.is_main_process:
            get_validation_loss(val_dataloader, num_frames, pose_guider_net, reference_net, video_net, vision_encoder, vae, vae_scaling_factor, device)


    # checkpoint models after stage one
    if accelerator.is_main_process:
        save_model_checkpoint(ckpt_dir, 2, pose_guider_net, reference_net, video_net, optimizer)

