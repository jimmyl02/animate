from os.path import join

import torch
from torchvision import io
from torchvision.transforms import v2
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor
from einops import rearrange, repeat
from accelerate import Accelerator
import torch.nn.functional as F

from models.poseguider import PoseGuider
from models.referencenet import ReferenceNet
from models.videonet import VideoNet
from util import get_models, retrieve_train_timesteps, get_initial_noise_latents, get_conditioning_embeddings, encode_images, retrieve_inference_timesteps

torch.manual_seed(17)

if __name__ == '__main__':
    ckpt_dir = '../ckpts'
    stage_one_batch_size, stage_two_batch_size = 1, 1
    stage_one_steps, stage_two_steps = 5000, 10000
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 10
    inference_steps = 50
    learning_rate = 1e-5

    # configuration_values
    config = {
        "gradient_accumulation_steps": 1,
        "mixed_precision": 'fp16'
    }

    # define the accelerator
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        log_with="wandb",
    )
    device = accelerator.device

    # get the models
    models = get_models(num_channels_latent, num_frames, device) #, ckpt='../ckpts/ckpt_s1_t1702357167.pt')
    
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

    scheduler.set_timesteps(inference_steps, device=device)
    video_net.batch_size = stage_one_batch_size

    # prepare data
    root_data_folder = '../datasets/TikTok_dataset/TikTok_dataset'
    ref_img_path = join(root_data_folder, '00001/images/0001.png')
    frame_list = [f'{i:04d}.png' for i in range(1, 11)]
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
        raw_img = io.read_image(join(root_data_folder, '00001', 'images', frame))

        poses.append(transform(pose_img))
        images.append(transform(raw_img))

    poses_tensor = torch.stack(poses).unsqueeze(0)
    images_tensor = torch.stack(images).unsqueeze(0)

    '''
    Begin stage one of training
    - stage one training objective is generating good images from poses (no temporal consistency)
    '''

    # define optimizer and prepare for stage one training
    optimizer = torch.optim.AdamW(
        list(pose_guider_net.parameters()) + list(video_net.parameters())  + list(reference_net.parameters()),
        lr=learning_rate,
    )
    vae, vision_encoder, pose_guider_net, reference_net, video_net, optimizer = accelerator.prepare(
        vae, vision_encoder, pose_guider_net, reference_net, video_net, optimizer
    )


    # statically set input and outputs for training

    # reshape data
    pose_data, raw_img_data, ref_img_data = poses_tensor.to(device), images_tensor.to(device), ref_img.to(device)
    pose_stack_data = rearrange(pose_data, 'b t c h w -> (b t) c h w')
    img_stack_data = rearrange(raw_img_data, 'b t c h w -> (b t) c h w')

    # generate noise latents
    initial_noise_shape = (stage_one_batch_size * num_frames, num_channels_latent, latent_height, latent_width)
    initial_noise_latents = get_initial_noise_latents(initial_noise_shape, scheduler, device)

    with torch.no_grad():
        clip_raw_img_embeddings, clip_raw_frame_embeddings, encoded_img_embeddings = get_conditioning_embeddings(vision_processor, vision_encoder, vae, ref_img_data, num_frames, vae_scaling_factor, device)
        img_latents = encode_images(vae, img_stack_data, vae_scaling_factor)

    stage_one_train, global_step = True, 0
    while stage_one_train:
        # train until global step is 30,000
        pose_guider_net.train()
        reference_net.train()
        video_net.train()

        with accelerator.accumulate(pose_guider_net, reference_net, video_net):
            # 3) generate pose guided images
            pose_embeddings = pose_guider_net(pose_stack_data)

            # 4) generate embeddings from reference net
            reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
            
            # reference_frame_embeddings is the reference embeddings repeated for each frame
            reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

            train_timesteps = retrieve_train_timesteps(scheduler, stage_one_batch_size * num_frames, device)

            # 5.3) generate conditioned noise
            conditioned_noise_latents = scheduler.add_noise(img_latents, initial_noise_latents, train_timesteps)
            conditioned_noise_latents = conditioned_noise_latents + pose_embeddings

            # 6) predict noise for video frames
            noise_pred = video_net(conditioned_noise_latents, train_timesteps, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
            loss = F.mse_loss(noise_pred, initial_noise_latents)
            accelerator.backward(loss)

            # backward optimizer pass
            torch.nn.utils.clip_grad_norm_(list(pose_guider_net.parameters()) + list(reference_net.parameters()) + list(video_net.parameters()),
                                        1.0)
            optimizer.step()
            optimizer.zero_grad()

            # log loss + update global step
            loss_item = loss.detach().item()
            print(f'step: {global_step} loss: {loss_item}')
            if global_step % 250 == 0:
                # infer and save image to train test
                with torch.no_grad():
                    # retrieve embeddings
                    pose_embeddings = pose_guider_net(pose_stack_data)
                    reference_embeddings = reference_net(encoded_img_embeddings, clip_raw_img_embeddings)
                    reference_frame_embeddings = [repeat(ref_emb, 'b c h w -> (b repeat) c h w', repeat=num_frames) for ref_emb in reference_embeddings]

                    latents = initial_noise_latents # fix the initial noise latents
                    timesteps = retrieve_inference_timesteps(scheduler, inference_steps, device)

                    for t in timesteps:
                        input_latents = latents + pose_embeddings
                        noise_pred = video_net(input_latents, t, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)
                        latents = scheduler.step(noise_pred, t, input_latents, return_dict=False)[0]

                    images = vae.decode(1 / vae_scaling_factor * latents, return_dict=False)[0]
                    images = vae_image_processor.postprocess(images, output_type="pil")
                    images[0].save(join('../out/train', f'train_{global_step}.png'))

            global_step += 1

            if global_step == stage_one_steps:
                stage_one_train = False
                break
