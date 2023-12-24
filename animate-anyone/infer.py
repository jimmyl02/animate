import shutil
from os.path import join

import torch
from torchvision import io
from torchvision.transforms import v2
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers.schedulers import PNDMScheduler
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from accelerate import Accelerator
from einops import repeat

from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet

from util import get_models, get_conditioning_embeddings, decode_images, get_initial_noise_latents, retrieve_inference_timesteps

device = 'cuda'
root_data_folder = '../datasets/TikTok_dataset/TikTok_dataset'
root_out_folder = '../out'
torch.manual_seed(17)

# infers the frames from the pose paths and ref image path then outputs to the out_dir
def infer_frames(ckpt, out_dir, pose_paths, ref_image_path):
    # configuration settings
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 8
    inference_steps = 50

    models = get_models(num_channels_latent, num_frames, device, ckpt=ckpt)
    print('[*] retrieved models')

    # get relevant models and settings from the models
    vision_processor: CLIPImageProcessor = models['vision_processor']
    vision_encoder: CLIPVisionModel = models['vision_encoder'].to('cuda:0', dtype=torch.bfloat16)
    vae: AutoencoderKL = models['vae'].to('cuda:0', dtype=torch.bfloat16)
    pose_guider_net: PoseGuider = models['pose_guider_net'].to('cuda:0')
    reference_net: ReferenceNet = models['reference_net'].to('cuda:0')
    video_net: VideoNet = models['video_net'].to('cuda:0')
    scheduler: PNDMScheduler = models['scheduler']
    vae_scaling_factor = vae.config.scaling_factor
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scaling_factor)

    scheduler.set_timesteps(inference_steps, device=device)
    video_net.batch_size = 1

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

    pose_guider_net, reference_net, video_net = accelerator.prepare(
        pose_guider_net, reference_net, video_net,
        device_placement=[False, False, False]
    )    

    # set all models to eval mode
    pose_guider_net.eval()
    reference_net.eval()
    video_net.eval()

    # load the data from the provided pose and ref image paths
    transform = v2.Compose([
            v2.Resize(384),
            v2.CenterCrop((576, 384)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    ref_img = io.read_image(ref_image_path)
    ref_img = transform(ref_img).to(device).unsqueeze(0)
    
    poses_list = []
    for pose_path in pose_paths:
        pose_img = io.read_image(pose_path)
        poses_list.append(transform(pose_img))
    poses = torch.stack(poses_list).to(device)

    # get conditioning embeddings
    clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_ref_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img, num_frames, vae_scaling_factor, device)

    with torch.no_grad():
        pose_embeddings = pose_guider_net(poses)
        ref_img_embeddings = reference_net(encoded_ref_img_embeddings, clip_raw_img_embeddings)
        reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in ref_img_embeddings]
        
        # create noise and timestep
        initial_noise_shape = (num_frames, num_channels_latent, latent_height, latent_width)
        latents = get_initial_noise_latents(initial_noise_shape, scheduler, device)
        timesteps = retrieve_inference_timesteps(scheduler, inference_steps, device)

        # complete denoising through all timesteps
        for i, t in enumerate(timesteps):
            print(f'[*] denoising step {i}/{inference_steps}')
            input_latents = latents + pose_embeddings
            noise_pred = video_net(input_latents, t, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
            latents = scheduler.step(noise_pred, t, input_latents, return_dict=False)[0]

        # decode image latents and write to out
        latents = latents.to(dtype=torch.bfloat16)
        images = vae.decode(1 / vae_scaling_factor * latents, return_dict=False)[0]
        images = vae_image_processor.postprocess(images, output_type="pil")
        for i in range(len(images)):
            # write out the decoded image
            image = images[i]
            image.save(join(out_dir, f'{i:04d}.png'))

            # write out the original pose image
            shutil.copy(pose_paths[i], join(out_dir, f'{i:04d}_pose.png'))

    return

if __name__ == '__main__':
    # infer frames
    ref_img_path = join(root_data_folder, '00052/images/0001.png')
    pose_paths = []
    pose_frames = [1,2,3,4,5,6,7,8]
    for pose_frame_num in pose_frames:
        pose_paths.append(join(root_data_folder, '00005/densepose', f'{pose_frame_num:04d}.png'))

    infer_frames('../ckpts/ckpt_s2_t1703365338.pt', join(root_out_folder, 'infer'), pose_paths, ref_img_path)

