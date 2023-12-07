import copy
import torch.nn as nn
from diffusers.models import UNet2DConditionModel

# ReferenceNet is a unet initialized from stable diffusion used to get reference conditions
class ReferenceNet(nn.Module):
    def __init__(self, sd_unet: UNet2DConditionModel):
        super(ReferenceNet, self).__init__()

        # create a deep copy of the sd_unet
        self.unet = copy.deepcopy(sd_unet)

        # create variable to store saved embeddings
        self._embeddings = []

        # hook the layers we want to get embeddings from in unet (this gives us gradients)
        for down_block in self.unet.down_blocks:
            # only save embedings for layers which have attentions
            if hasattr(down_block, 'attentions'):
                for resnet in down_block.resnets:
                    resnet.register_forward_hook(self.save_output_embedding())

        # exclude last resnet as mid_block doesn't attend to last output
        for resnet in self.unet.mid_block.resnets[:-1]:
            resnet.register_forward_hook(self.save_output_embedding())

        for up_block in self.unet.up_blocks:
            # only save embedings for layers which have attentions
            if hasattr(up_block, 'attentions'):
                for resnet in up_block.resnets:
                    resnet.register_forward_hook(self.save_output_embedding())

    # define hook to store embeddings
    def save_output_embedding(self):
        def fn(_, __, output):
            self._embeddings.append(output)

        return fn

    # forward pass just passes pose + conditioning embeddings to unet and returns activations
    def forward(self, pose_embeddings, clip_condition_embeddings):
        # clear the embeddings that have been saved
        self._embeddings = []

        # forward pass the pose + conditioning embeddings through the unet
        self.unet(
            pose_embeddings,
            1,
            encoder_hidden_states=clip_condition_embeddings,
        )

        # return the embeddings which have the stored activations
        return self._embeddings
