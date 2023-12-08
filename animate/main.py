import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from accelerate import Accelerator
import torch.nn.functional as F

from datasets.jafarin import JafarinVideoDataset
from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(17)

def debug():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

    prompt = 'Astronaut riding a horse on desert, holding a Neo matrix indices flag, matrix theme, artstation, concept art, crepuscular rays, smooth, sharp focus, hd'
    img = pipe(prompt).images[0]
    img.save('tmp.png')


# get_models gets the default models
def get_models(latent_channels, num_frames):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    
    # define and freeze vision encoder weights
    vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    for param in vision_encoder.parameters():
        param.requires_grad = False

    scheduler = pipe.scheduler
    reference_net = ReferenceNet(pipe.unet).to(device)
    video_net = VideoNet(pipe.unet, num_frames=num_frames).to(device)
    pose_guider_net = PoseGuider(latent_channels).to(device)

    # define and freeze vae weights
    vae = pipe.vae
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
        0, scheduler.config.num_train_timesteps, (batch_size,), device=device,
        dtype=torch.int64,
    )

    return timesteps

# get_initial_inference_noise_latents gets the initial noise latents
def get_initial_inference_noise_latents(shape, scheduler, device, dtype):
    initial_noise_latents = torch.randn(shape, device=device, dtype=dtype)

    # scale the noise latents by the scheduler
    initial_noise_latents = initial_noise_latents * scheduler.init_noise_sigma

    return initial_noise_latents

# get_initial_train_noise_latents gets the initial noise latents
def get_initial_train_noise_latents(shape, device, dtype):
    initial_noise_latents = torch.randn(shape, device=device, dtype=dtype)

    return initial_noise_latents

# gets the dataloader for the image dataset
def get_image_dataloader(batch_size, num_frames):
    dataset = JafarinVideoDataset('../datasets/TikTok_dataset/TikTok_dataset', num_frames, device)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader

if __name__ == '__main__':
    stage_one_batch_size = 2
    stage_two_batch_size = 1
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 16
    inference_steps = 50

    lr_warmup_steups = 500
    learning_rate = 1e-4

    # configuration_values
    config = {
        "gradient_accumulation_steps": 2,
        "mixed_precision": 'fp16'
    }

    # get the models
    models = get_models(num_channels_latent, num_frames)
    
    # get relevant models and settings from the models
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
    
    # get the image data loader
    dataloader = get_image_dataloader(stage_one_batch_size, num_frames)

    # define accelerator, dataloader, and optimizer for stage one training
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with="wandb",
    )

    if accelerator.is_main_process:
        # accelerator.init_trackers("animate")
        pass

    optimizer = torch.optim.AdamW(
        list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()), 
        lr=learning_rate,
    )

    # vae, pose_guider_net, reference_net, video_net, optimizer, dataloader = accelerator.prepare(
    #     vae, pose_guider_net, reference_net, video_net, optimizer, dataloader
    # )

    # DEBUG - perform one loop training loop for the first stage of training
    idx_data = next(enumerate(dataloader))
    data = idx_data[1]
    pose_data, raw_img_data, ref_img_data = data[0], data[1], data[2]

    # we stack the pose to batch it through poseguider
    pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')

    # 1) generate embeddings from CLIP vision encoder
    with torch.no_grad():
        processed_img_embeddings = models['vision_processor'](images=ref_img_data, return_tensors="pt").to(device)
        clip_raw_img_embeddings = models['vision_encoder'](**processed_img_embeddings).last_hidden_state.to(dtype=torch.float16)
    
    # clip_raw_frame_embeddings is the raw clip embeddings repeated for each frame
    clip_raw_frame_embeddings = repeat(clip_raw_img_embeddings, 'b l d -> (b repeat) l d', repeat=num_frames)

    # 2) generate pose guided images
    pose_embeddings = pose_guider_net(pose_stack_data).to(dtype=torch.float16)

    # 3) encode reference image with the vae encoder
    with torch.no_grad():
        half_precision_ref_img_data = ref_img_data.to(dtype=torch.float16)
        encoded_img_embeddings = vae.encode(half_precision_ref_img_data)['latent_dist'].mean * vae_scaling_factor

    # 4) generate embeddings from reference net
    reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
    
    # reference_frame_embeddings is the reference embeddings repeated for each frame
    reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

    # 5) generate the noisy latents
    initial_noise_shape = (stage_one_batch_size * num_frames, num_channels_latent, latent_height, latent_width)
    initial_noise_latents = get_initial_train_noise_latents(initial_noise_shape, device=device, dtype=torch.float16)

    # 5.1) generate train timesteps
    train_timesteps = retrieve_train_timesteps(scheduler, stage_one_batch_size * num_frames, device)

    # 5.2) generate conditioned noise
    conditioned_noise_latents = scheduler.add_noise(pose_embeddings, initial_noise_latents, train_timesteps)

    # 6) predict noise for video frames
    print('debug - predicting noise for video frames')
    noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
    loss = F.mse_loss(noise_pred, initial_noise_latents)
    # accelerator.backward(loss)
    loss.backward()

    # TODO(jimmy): consider clipping gradient norms
    optimizer.step()
    optimizer.zero_grad()

    # decode latents from initial latents, noise prediction, and the timesteps
    # print('debug - latents', noise_pred.shape, train_timesteps.shape, initial_noise_latents.shape)
    # latents = scheduler.step(noise_pred, train_timesteps.cpu(), initial_noise_latents, return_dict=False)[0]

    # 7) stage two training objective is generating temporally consistent images (freeze video net besides temporal)

    # 8) inference stage - run main video denoising loop, with condition embeddings from reference net + CLIP
    # TODO(jimmy): write the main unet denoising loop
    
