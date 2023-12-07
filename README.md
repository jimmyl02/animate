# animate

this repository aims to reproduce [animate anyone](https://arxiv.org/pdf/2311.17117.pdf) which enables avatars to be produced in a consistent video with poses as the guiding prior

## dataset

the primary dataset is [Jafarian's tiktok dataset](https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset/) which contains tiktok dance videos along with pose extractions.

## architecture

the architecture is directly from animate anyone and uses stable diffusion 1.5 as the core unet architecture

## training process

initializations - pose guider (gaussian except zero convolution)

step 1 - exclude temporal layer in video net, freeze vae encoder-decoder and clip encoder, train with single frame including reference net and pose guider to denoise the target image
step 2 - introduce temporal layer in video net, fix weights for rest of the network

## notes

stability training run - adanw, 2048 batch, lr warmup 10,000 steps then constant
