#!/bin/bash
python src/inference.py --input_video_path "../datasets/movie/test_clip/woz_test_clip.mp4" \
        --pretrained_sd_dir "../ckpts/stable-diffusion-v1-5" \
        --video_outpainting_model_dir "../ckpts/m3ddm-base" \
        --output_dir "./out" \
        --target_ratio_list "1:1" \
        --copy_original
