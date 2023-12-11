import time
from os.path import join

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from accelerate import Accelerator
import torch.nn.functional as F

from datasets.jafarin import JafarinVideoDataset
from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet

torch.manual_seed(17)

def debug():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

    prompt = 'Astronaut riding a horse on desert, holding a Neo matrix indices flag, matrix theme, artstation, concept art, crepuscular rays, smooth, sharp focus, hd'
    img = pipe(prompt).images[0]
    img.save('tmp.png')

# save_model_checkpoint saves the model as a checkpoint
def save_model_checkpoint(ckpt_dir, stage, pose_guider_net, reference_net, video_net, optimizer):
    states = {
        'pose_guider_net': pose_guider_net.state_dict(),
        'reference_net': reference_net.state_dict(),
        'video_net': video_net.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(states, join(ckpt_dir, f'ckpt_s{stage}_t{int(time.time())}.pt'))

# get_models gets the default models
def get_models(latent_channels, num_frames, device, ckpt:str=""):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    
    # define and freeze vision encoder weights
    vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False

    scheduler = pipe.scheduler
    reference_net = ReferenceNet(pipe.unet).to(device)
    video_net = VideoNet(pipe.unet, num_frames=num_frames).to(device)
    pose_guider_net = PoseGuider(latent_channels).to(device)

    # define and freeze vae weights
    vae = pipe.vae
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    models = {
        'scheduler': scheduler,
        'vae': vae,
        'unet': pipe.unet,
        'vision_processor': vision_processor,
        'vision_encoder': vision_encoder,
        'reference_net': reference_net,
        'video_net': video_net,
        'pose_guider_net': pose_guider_net
    }

    return models


# retrieve_inference_timesteps retrieves timesteps for diffusion process
def retrieve_inference_timesteps(
    scheduler,
    num_inference_steps,
    device,
):
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    return timesteps

# retrieve_train_timesteps retrieves timesteps for training diffusion process
def retrieve_train_timesteps(
        scheduler: PNDMScheduler,
        batch_size,
        device
):
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (batch_size,), dtype=torch.int64, device=device
    )

    return timesteps

# get_initial_inference_noise_latents gets the initial noise latents
def get_initial_inference_noise_latents(shape, scheduler, device):
    initial_noise_latents = torch.randn(shape, device=device)

    # scale the noise latents by the scheduler
    initial_noise_latents = initial_noise_latents * scheduler.init_noise_sigma

    return initial_noise_latents

# get_initial_train_noise_latents gets the initial noise latents
def get_initial_train_noise_latents(shape, device):
    initial_noise_latents = torch.randn(shape, device=device)

    return initial_noise_latents

# gets the dataloader for the image dataset
def get_image_dataloader(batch_size, num_frames):
    dataset = JafarinVideoDataset('../datasets/TikTok_dataset/TikTok_dataset', num_frames)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [.97, .03])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

# get_conditioning_embeddings gets the condition embeddings used to start the process
def get_conditioning_embeddings(vision_encoder, vae, ref_img_data, device):
    with torch.no_grad():
        # 1) generate embeddings from CLIP vision encoder
        processed_img_embeddings = vision_processor(images=ref_img_data, return_tensors="pt").to(device)
        clip_raw_img_embeddings = vision_encoder(**processed_img_embeddings).last_hidden_state
    
        # clip_raw_frame_embeddings is the raw clip embeddings repeated for each frame
        clip_raw_frame_embeddings = repeat(clip_raw_img_embeddings, 'b l d -> (b repeat) l d', repeat=num_frames)

        # 2) encode reference image with the vae encoder
        encoded_img_embeddings = encode_images(vae, ref_img_data)

    return clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings

def encode_images(vae, image_data):
    with torch.no_grad():
        encoded_img_embeddings = vae.encode(image_data)['latent_dist'].mean * vae_scaling_factor
        return encoded_img_embeddings

# get_validation_loss gets the loss of the models on the validation set
def get_validation_loss(val_dataloader, num_frames, pose_guider_net: PoseGuider, reference_net: ReferenceNet, video_net: VideoNet, vision_encoder, vae, device):
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
        clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_encoder, vae, ref_img_data, device)

        # forward pass with no gradients and calculate loss
        with torch.no_grad():
            # 3) generate pose guided images
            pose_embeddings = pose_guider_net(pose_stack_data)

            # 4) generate embeddings from reference net
            reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
            
            # reference_frame_embeddings is the reference embeddings repeated for each frame
            reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

            # 5) generate the noisy latents
            initial_noise_shape = (stage_one_batch_size * num_frames, num_channels_latent, latent_height, latent_width)
            initial_noise_latents = get_initial_train_noise_latents(initial_noise_shape, device)

            # 5.1) generate train timesteps
            train_timesteps = retrieve_train_timesteps(scheduler, stage_one_batch_size * num_frames, device)

            # 5.2) generate image latents
            img_latents = encode_images(vae, img_stack_data)

            # 5.3) generate conditioned noise
            conditioned_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)
            conditioned_noise_latents = conditioned_noise_latents + pose_embeddings

            # 6) predict noise for video frames
            noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
            loss = F.mse_loss(noise_pred, initial_noise_latents)
            loss_item = loss.detach().item()
            total_loss += loss_item
            total_items += 1

    # after finding the average of entire validation set
    val_loss = total_loss / total_items
    accelerator.log({"stage_one_val_loss": val_loss})
    print(f'val loss: {val_loss}')

if __name__ == '__main__':
    ckpt_dir = '../ckpts'
    stage_one_batch_size, stage_two_batch_size = 1, 1
    stage_one_steps, stage_two_steps = 30000, 10000
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 10
    inference_steps = 50

    lr_warmup_steps = 500
    learning_rate = 1e-5

    # configuration_values
    config = {
        "gradient_accumulation_steps": 2,
        "mixed_precision": 'fp16'
    }

    # define the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        log_with="wandb",
    )
    device = accelerator.device
    # if accelerator.is_main_process:
    #     accelerator.init_trackers(
    #         project_name="animate"
    #     )

    # get the models
    models = get_models(num_channels_latent, num_frames, device)
    
    # get relevant models and settings from the models
    vision_processor: CLIPImageProcessor = models['vision_processor']
    vision_encoder: CLIPVisionModel = models['vision_encoder']
    vae: AutoencoderKL = models['vae']
    pose_guider_net: PoseGuider = models['pose_guider_net']
    reference_net: ReferenceNet = models['reference_net']
    video_net: VideoNet = models['video_net']
    scheduler: PNDMScheduler = models['scheduler']
    vae_scaling_factor = vae.config.scaling_factor

    scheduler.set_timesteps(inference_steps, device=device)
    video_net.batch_size = stage_one_batch_size

    '''
    Begin stage one of training
    - stage one training objective is generating good images from poses (no temporal consistency)
    '''

    # define dataloader and optimizer for stage one training
    train_dataloader, val_dataloader = get_image_dataloader(stage_one_batch_size, num_frames)
    optimizer = torch.optim.AdamW(
        list(pose_guider_net.parameters()) + list(video_net.parameters())  + list(reference_net.parameters()),
        lr=learning_rate,
    )
    vae, vision_encoder, pose_guider_net, reference_net, video_net, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        vae, vision_encoder, pose_guider_net, reference_net, video_net, optimizer, train_dataloader, val_dataloader
    )

    # stage_one_train, global_step = True, 0
    # while stage_one_train:
    #     # train until global step is 30,000
    #     pose_guider_net.train()
    #     reference_net.train()
    #     video_net.train()
    #     for step, data in enumerate(train_dataloader):
    #         if data[0].shape[1] == 0:
    #             # we failed to load the data, just continue to next data batch
    #             continue

    #         pose_data, raw_img_data, ref_img_data = data[0].to(device), data[1].to(device), data[2].to(device)

    #         # we stack the pose and images to batch it through poseguider
    #         pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
    #         img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

    #         # get conditioning embeddings
    #         clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_encoder, vae, ref_img_data, device)

    #         with accelerator.accumulate(pose_guider_net, reference_net, video_net):
    #             # 3) generate pose guided images
    #             pose_embeddings = pose_guider_net(pose_stack_data)

    #             # 4) generate embeddings from reference net
    #             reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
                
    #             # reference_frame_embeddings is the reference embeddings repeated for each frame
    #             reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

    #             # 5) generate the noisy latents
    #             initial_noise_shape = (stage_one_batch_size * num_frames, num_channels_latent, latent_height, latent_width)
    #             initial_noise_latents = get_initial_train_noise_latents(initial_noise_shape, device)

    #             # 5.1) generate train timesteps
    #             train_timesteps = retrieve_train_timesteps(scheduler, stage_one_batch_size * num_frames, device)

    #             # 5.2) generate image latents
    #             img_latents = encode_images(vae, img_stack_data)

    #             # 5.3) generate conditioned noise
    #             conditioned_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)
    #             conditioned_noise_latents = conditioned_noise_latents + pose_embeddings

    #             # 6) predict noise for video frames
    #             noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
    #             loss = F.mse_loss(noise_pred, initial_noise_latents)
    #             accelerator.backward(loss)

    #             # backward optimizer pass
    #             torch.nn.utils.clip_grad_norm_(list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
    #                                         1.0)
    #             optimizer.step()
    #             optimizer.zero_grad()

    #         # log loss + update global step
    #         loss_item = loss.detach().item()
    #         accelerator.log({"stage_one_train_loss": loss_item})
    #         #if global_step % 500 == 0:
    #         print(f'step: {global_step} loss: {loss_item}')
    #         global_step += 1

    #         if global_step == stage_one_steps:
    #             stage_one_train = False
    #             break

    #     # evaluate loss on the validation set
    #     get_validation_loss(val_dataloader, num_frames, pose_guider_net, reference_net, video_net, vision_encoder, vae, device)


    # # checkpoint models after stage one
    # save_model_checkpoint(ckpt_dir, 1, pose_guider_net, reference_net, video_net, optimizer)
    '''
    Begin stage two of training
    - stage two training objective is generating temporally consistent images (freeze video net besides temporal)
    '''

    # freeze all model weights in pose guider, reference_net, video_net
    for param in pose_guider_net.parameters():
        param.requires_grad = False
    for param in reference_net.parameters():
        param.requires_grad = False
    for param in video_net.parameters():
        param.requires_grad = False

    pose_guider_net.eval()
    reference_net.eval()

    # unfreeze only temporal layers
    for ref_cond_block in video_net.ref_cond_attn_blocks:
        for param in ref_cond_block.tam.parameters():
            param.requires_grad = True

    # prepare stage two optimizer
    s2_optimizer = torch.optim.AdamW(
        video_net.parameters(),
        lr=learning_rate,
    )
    s2_optimizer = accelerator.prepare(s2_optimizer)

    stage_two_train, global_step = True, 0
    while stage_two_train:
        # train until global step is 10,000
        video_net.train()
        for step, data in enumerate(train_dataloader):
            if data[0].shape[1] == 0:
                # we failed to load the data, just continue to next data batch
                continue

            pose_data, raw_img_data, ref_img_data = data[0].to(device), data[1].to(device), data[2].to(device)

            # we stack the pose and images to batch it through poseguider
            pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
            img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

            # get conditioning embeddings
            clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_encoder, vae, ref_img_data, device)

            with accelerator.accumulate(pose_guider_net, reference_net, video_net):
                # 3) generate pose guided images
                pose_embeddings = pose_guider_net(pose_stack_data)

                # 4) generate embeddings from reference net
                reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
                
                # reference_frame_embeddings is the reference embeddings repeated for each frame
                reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

                # 5) generate the noisy latents
                initial_noise_shape = (stage_one_batch_size * num_frames, num_channels_latent, latent_height, latent_width)
                initial_noise_latents = get_initial_train_noise_latents(initial_noise_shape, device)

                # 5.1) generate train timesteps
                train_timesteps = retrieve_train_timesteps(scheduler, stage_one_batch_size * num_frames, device)

                # 5.2) generate image latents
                img_latents = encode_images(vae, img_stack_data)

                # 5.3) generate conditioned noise
                conditioned_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)
                conditioned_noise_latents = conditioned_noise_latents + pose_embeddings

                # 6) predict noise for video frames
                noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=False)
                loss = F.mse_loss(noise_pred, initial_noise_latents)
                accelerator.backward(loss)

                # backward optimizer pass
                torch.nn.utils.clip_grad_norm_(list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
                                            1.0)
                s2_optimizer.step()
                s2_optimizer.zero_grad()

            # log loss + update global step
            loss_item = loss.detach().item()
            accelerator.log({"stage_two_train_loss": loss_item})
            #if global_step % 500 == 0:
            print(f'step: {global_step} loss: {loss_item}')
            global_step += 1

            if global_step == stage_one_steps:
                stage_one_train = False
                break


    # 8) inference stage - run main video denoising loop, with condition embeddings from reference net + CLIP
    # TODO(jimmy): write the main unet denoising loop
    
