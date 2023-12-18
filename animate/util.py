import gc
import copy
from os.path import join

import torch
from torchvision import io
from torchvision.transforms import v2
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers.schedulers import PNDMScheduler
from einops import rearrange, repeat

from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet


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
def get_initial_noise_latents(shape, scheduler, device):
    initial_noise_latents = torch.randn(shape, device=device)

    # scale the noise latents by the scheduler
    initial_noise_latents = initial_noise_latents * scheduler.init_noise_sigma

    return initial_noise_latents

# get_conditioning_embeddings gets the condition embeddings used to start the process
def get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img_data, num_frames, vae_scaling_factor, device):
    with torch.no_grad():
        # 1) generate embeddings from CLIP vision encoder
        processed_ref_img_embeddings = vision_processor(images=ref_img_data, return_tensors="pt").to(device)
        clip_raw_img_embeddings = vision_encoder(**processed_ref_img_embeddings).last_hidden_state
        clip_raw_img_embeddings = clip_raw_img_embeddings / clip_raw_img_embeddings.norm(p=2, dim=-1, keepdim=True) # we need to normalize the CLIP embedding
    
        # clip_raw_frame_embeddings is the raw clip embeddings repeated for each frame
        clip_raw_frame_embeddings = repeat(clip_raw_img_embeddings, 'b l d -> (b repeat) l d', repeat=num_frames)

        # 2) encode reference image with the vae encoder
        encoded_ref_img_embeddings = encode_images(vae, ref_img_data, vae_scaling_factor)

    return clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_ref_img_embeddings

# encode_images encodes images with the vae
def encode_images(vae, image_data, vae_scaling_factor, dtype=torch.bfloat16):
    with torch.no_grad():
        encoded_img_embeddings = vae.encode(image_data.to(dtype=dtype))['latent_dist'].mean * vae_scaling_factor
        return encoded_img_embeddings

# decode_images decodes images with the vae
def decode_images(vae, encoded_img_data, vae_scaling_factor):
    with torch.no_grad():
        return vae.decode(1 / vae_scaling_factor * encoded_img_data)[0]

# get_models gets the default models
def get_models(latent_channels, num_frames, device, ckpt: str=""):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    
    # define and freeze vision encoder weights
    vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vision_encoder.requires_grad_(False)
    vision_encoder.eval()

    scheduler = copy.deepcopy(pipe.scheduler)
    reference_net = ReferenceNet(pipe.unet).to(device)
    video_net = VideoNet(pipe.unet, num_frames=num_frames).to(device)
    pose_guider_net = PoseGuider(latent_channels).to(device)

    # define and freeze vae weights
    vae = copy.deepcopy(pipe.vae)
    vae.requires_grad_(False)
    vae.eval()

    # load models from ckpt if given
    if ckpt != "":
        ckpt_dict = torch.load(ckpt)
        pose_guider_net.load_state_dict(ckpt_dict['pose_guider_net'])
        reference_net.load_state_dict(ckpt_dict['reference_net'])
        video_net.load_state_dict(ckpt_dict['video_net'])

    # cleanup the pipe which removes excess memory from unused unet
    del pipe
    gc.collect()
    torch.cuda.empty_cache() 

    # package the models and return to requester
    models = {
        'scheduler': scheduler,
        'vae': vae,
        'vision_processor': vision_processor,
        'vision_encoder': vision_encoder,
        'reference_net': reference_net,
        'video_net': video_net,
        'pose_guider_net': pose_guider_net
    }

    return models

# infer_fixed_sample runs inference for a fixed sample
def infer_fixed_sample(noise_shape, num_frames, inference_steps, scheduler, vision_processor,
                        vision_encoder, vae, vae_scaling_factor, pose_guider_net, reference_net,
                        video_net, vae_image_processor, out_path, device, dtype=torch.bfloat16):
    # statically set input and outputs for training image generation
    root_data_folder = '../datasets/TikTok_dataset/TikTok_dataset'
    ref_img_path = join(root_data_folder, '00001/images/0001.png')
    frame_list = [f'{i:04d}.png' for i in range(1, num_frames+1)]
    transform = v2.Compose([
            v2.Resize(384),
            v2.CenterCrop((576, 384)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    ref_img = io.read_image(ref_img_path)
    ref_img = transform(ref_img).to(device).unsqueeze(0)

    poses, images = [], []
    for frame in frame_list:
        pose_img = io.read_image(join(root_data_folder, '00001', 'densepose', frame))
        poses.append(transform(pose_img))

    poses_tensor = torch.stack(poses).to(device).unsqueeze(0)
    pose_stack_data = rearrange(poses_tensor, 'b t c h w -> (b t) c h w')

    with torch.no_grad():
        clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img, num_frames, vae_scaling_factor, device)

        # retrieve embeddings
        pose_embeddings = pose_guider_net(pose_stack_data)
        reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
        reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

        latents = get_initial_noise_latents(noise_shape, scheduler, device)
        timesteps = retrieve_inference_timesteps(scheduler, inference_steps, device)

        for t in timesteps:
            input_latents = latents + pose_embeddings
            noise_pred = video_net(input_latents, t, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
            latents = scheduler.step(noise_pred, t, input_latents, return_dict=False)[0]

    images = vae.decode(1 / vae_scaling_factor * latents.to(dtype=dtype), return_dict=False)[0]
    images = vae_image_processor.postprocess(images, output_type="pil")
    images[0].save(out_path)

# infer_fixed_sample runs inference for a fixed sample
def infer_fixed_sample_mp(noise_shape, num_frames, inference_steps, scheduler, vision_processor,
                        vision_encoder, vae, vae_scaling_factor, pose_guider_net, reference_net,
                        video_net, vae_image_processor, out_path, dtype=torch.bfloat16):
    # statically set input and outputs for training image generation
    root_data_folder = '../datasets/TikTok_dataset/TikTok_dataset'
    ref_img_path = join(root_data_folder, '00001/images/0001.png')
    frame_list = [f'{i:04d}.png' for i in range(1, num_frames+1)]
    transform = v2.Compose([
            v2.Resize(384),
            v2.CenterCrop((576, 384)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    ref_img = io.read_image(ref_img_path)
    ref_img = transform(ref_img).to('cuda:0').unsqueeze(0)

    poses, images = [], []
    for frame in frame_list:
        pose_img = io.read_image(join(root_data_folder, '00001', 'densepose', frame))
        poses.append(transform(pose_img))

    poses_tensor = torch.stack(poses).to('cuda:0').unsqueeze(0)
    pose_stack_data = rearrange(poses_tensor, 'b t c h w -> (b t) c h w')

    with torch.no_grad():
        clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img, num_frames, vae_scaling_factor, 'cuda:0')
        clip_raw_frame_embeddings = clip_raw_frame_embeddings.to('cuda:1')

        # retrieve embeddings
        pose_embeddings = pose_guider_net(pose_stack_data).to('cuda:1')
        reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
        reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames).to('cuda:1') for ref_emb in reference_embeddings]

        latents = get_initial_noise_latents(noise_shape, scheduler, 'cuda:1')
        timesteps = retrieve_inference_timesteps(scheduler, inference_steps, 'cuda:1')

        for t in timesteps:
            input_latents = latents + pose_embeddings
            noise_pred = video_net(input_latents, t, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
            latents = scheduler.step(noise_pred, t, input_latents, return_dict=False)[0]

    images = vae.decode(1 / vae_scaling_factor * latents.to('cuda:0', dtype=dtype), return_dict=False)[0]
    images = vae_image_processor.postprocess(images, output_type="pil")
    images[0].save(out_path)

