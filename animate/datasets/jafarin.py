import os
from os.path import join
from typing import Any
import random

import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2

class JafarinVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames) -> None:
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.data_files = [f for f in os.listdir(root_dir)]

        # create transforms
        # want to format all images to be 384 x 576
        self.transform = v2.Compose([
            v2.Resize(384),
            v2.CenterCrop((576, 384)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index) -> Any:
        data_file = self.data_files[index]
        data_folder = join(self.root_dir, data_file)
        frame_list = [f for f in os.listdir(join(data_folder, 'images'))]
        frame_list.sort()

        # randomly sample only num_frames of this sequence
        # NOTE: we should ensure that there are enough frames to use
        max_start_idx = len(frame_list) - self.num_frames
        start_idx = random.randint(0, max_start_idx)
        frame_list = frame_list[start_idx:start_idx+self.num_frames]

        try:
            # in happy path, return images
            ref_img = io.read_image(join(data_folder, 'images', random.choice(frame_list)))
            ref_img = self.transform(ref_img)

            poses, images = [], []
            for frame in frame_list:
                pose_img = io.read_image(join(data_folder, 'densepose', frame))
                raw_img = io.read_image(join(data_folder, 'images', frame))

                poses.append(self.transform(pose_img))
                images.append(self.transform(raw_img))

            poses_tensor = torch.stack(poses)
            images_tensor = torch.stack(images)

            return poses_tensor, images_tensor, ref_img
        except Exception as e:
            # in error path, return empty data
            print('[*] failed to get item from dataset', e)

            # hack to pass the dataloader through
            return torch.empty((0,0)), torch.empty((0,0)), torch.empty((0,0))
