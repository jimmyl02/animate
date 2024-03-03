import os
from os.path import join
from typing import Any
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2
from transformers import CLIPImageProcessor


class JafarinImageDataset(Dataset):
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.data_files = [f for f in os.listdir(root_dir)]
        self.clip_image_processor = CLIPImageProcessor()

        # create transforms
        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    (576, 384),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.ToTensor(),
                v2.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    (576, 384),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.ToTensor(),
            ]
        )

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index) -> Any:
        data_file = self.data_files[index]
        data_folder = join(self.root_dir, data_file)
        frame_list = [f for f in os.listdir(join(data_folder, 'images'))]
        frame_list.sort()

        while True:
            try:
                # in happy path, return images
                ref_img = Image.open(join(data_folder, 'images', random.choice(frame_list)))

                frame_idx = random.randrange(0, len(frame_list))
                pose_img = Image.open(join(data_folder, 'densepose', frame_list[frame_idx]))
                tgt_img = Image.open(join(data_folder, 'images', frame_list[frame_idx]))

                state = torch.get_rng_state()
                tgt_img = self.augmentation(tgt_img, self.transform, state)
                tgt_pose_img = self.augmentation(pose_img, self.cond_transform, state)
                ref_img_vae = self.augmentation(ref_img, self.transform, state)
                clip_image = self.clip_image_processor(
                    images=ref_img, return_tensors="pt"
                ).pixel_values[0]

                sample = dict(
                    img=tgt_img,
                    tgt_pose=tgt_pose_img,
                    ref_img=ref_img_vae,
                    clip_images=clip_image,
                )

                return sample
            except Exception as e:
                # in error path, retry with new random index
                print('[*] failed to get item from dataset', e)


class JafarinVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames) -> None:
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.data_files = [f for f in os.listdir(root_dir)]

        # create transforms
        # want to format all images to be 384 x 576
        self.transform = v2.Compose([
            # v2.Resize(384, interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
            v2.RandomResizedCrop((576, 384), interpolation=v2.InterpolationMode.BILINEAR, antialias=None),
            v2.ToDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index) -> Any:
        data_file = self.data_files[index]
        data_folder = join(self.root_dir, data_file)
        frame_list = [f for f in os.listdir(join(data_folder, 'images'))]
        frame_list.sort()

        while True:
            # randomly sample only num_frames of this sequence
            # NOTE: we should ensure that there are enough frames to use
            max_start_idx = len(frame_list) - self.num_frames
            start_idx = random.randint(0, max_start_idx)
            ret_frame_list = frame_list[start_idx:start_idx+self.num_frames]

            try:
                # in happy path, return images
                ref_img = io.read_image(join(data_folder, 'images', random.choice(frame_list)))
                ref_img = self.transform(ref_img)

                poses, images = [], []
                for frame in ret_frame_list:
                    pose_img = io.read_image(join(data_folder, 'densepose', frame))
                    raw_img = io.read_image(join(data_folder, 'images', frame))

                    poses.append(self.transform(pose_img))
                    images.append(self.transform(raw_img))

                poses_tensor = torch.stack(poses)
                images_tensor = torch.stack(images)

                return poses_tensor, images_tensor, ref_img
            except Exception as e:
                # in error path, retry with new random index
                print('[*] failed to get item from dataset', e)
