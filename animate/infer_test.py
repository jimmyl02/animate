import shutil
from os.path import join

import torch
from torchvision import io
from torchvision.transforms import v2
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import PNDMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor

from util import get_initial_noise_latents, retrieve_inference_timesteps

device = 'cuda'
root_data_folder = '../datasets/TikTok_dataset/TikTok_dataset'
root_out_folder = '../out'
torch.manual_seed(17)

if __name__=='__main__':
    ref_img_path = join(root_data_folder, '00001/images/0001.png')
    out_dir = join(root_out_folder, 'test')

    # configuration settings
    latent_width, latent_height = 48, 72
    num_channels_latent = 4
    num_frames = 1
    inference_steps = 100
    guidance_scale = 7.5

    # get relevant models and settings from the models
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False

    print('[*] retrieved models')

    vae: AutoencoderKL = pipe.vae
    unet: UNet2DConditionModel = pipe.unet
    scheduler: PNDMScheduler = pipe.scheduler
    vae_scaling_factor = vae.config.scaling_factor
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scaling_factor)

    scheduler.set_timesteps(inference_steps, device=device)

    # we extract the unet from uninitialized video net and attempt to generate some random frames
    transform = v2.Compose([
            v2.Resize(384),
            v2.CenterCrop((576, 384)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    ref_img = io.read_image(ref_img_path)
    ref_img = transform(ref_img).to(device).unsqueeze(0)

    # clip ref img embedding
    processed_ref_img = vision_processor(images=ref_img, return_tensors="pt").to(device)
    ref_img_embedding = vision_encoder(**processed_ref_img).last_hidden_state
    ref_img_embedding = ref_img_embedding / ref_img_embedding.norm(p=2, dim=-1, keepdim=True) # we need to normalize the CLIP embedding

    # debug - use text prompt and encoder for now
    text_prompt = "fireworks in the night sky over the golden gate bridge, hyper-realistic, dark sky, midnight"
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    with torch.no_grad():
        text_inputs = tokenizer(
            text_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(text_inputs.input_ids, attention_mask=None)[0].float()
        # neg_text_inputs = tokenizer(
        #     "",
        #     padding="max_length",
        #     max_length=tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # ).to(device)
        # neg_text_embeds = text_encoder(neg_text_inputs.input_ids, attention_mask=None)[0]

    # text_embeddings, neg_text_embeds = pipe.encode_prompt(text_prompt, device, 1, True)
    # full_text_embeds=torch.cat([neg_text_embeds, text_embeddings])
    # print(neg_text_embeds.shape, ref_img_embedding.shape)
    # full_text_embeds=torch.cat([neg_text_embeds, ref_img_embedding])

    # create noise and timestep
    initial_noise_shape = (num_frames, num_channels_latent, 64, 64)
    latents = get_initial_noise_latents(initial_noise_shape, scheduler, device)
    timesteps = retrieve_inference_timesteps(scheduler, inference_steps, device)

    # test pipe
    # pipe(text_prompt, latents=latents).images[0].save('../out/test/real.png')
    pipe(prompt_embeds=text_embeddings, num_inference_steps=inference_steps, text_encoder=None, guidance_scale=1, latents=latents).images[0].save('../out/test/real.png')

    with torch.no_grad():
        # complete denoising through all timesteps
        for i, t in enumerate(timesteps):
            print(f'[*] denoising step {i}/{inference_steps}')
            # duplicate latents to prepare for classifier free guidance
            # latent_model_input = torch.cat([latents] * 2)
            # noise_pred = unet(latent_model_input, t, encoder_hidden_states=full_text_embeds)[0]
            # noise_pred = unet(latents, t, encoder_hidden_states=ref_img_embedding)[0]
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings)[0]

            # perform classifier free guidance
            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # decode images
        images = vae.decode(1 / vae_scaling_factor * latents, return_dict=False)[
                0
            ]
        images = vae_image_processor.postprocess(images, output_type="pil")
        images[0].save(join(out_dir, 'test.png'))
