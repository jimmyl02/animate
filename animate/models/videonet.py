import copy
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel, Transformer2DModel
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SpatialAttentionModule is a spatial attention module
# credit: https://github.com/Peachypie98/CBAM/blob/main/cbam.py
class SpatialAttentionModule(nn.Module):
    def __init__(self) -> None:
        super(SpatialAttentionModule, self).__init__()

        # define covolution to find attention maps
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1)

    # forward passes the activation through a spatial attention module
    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 


# TemporalAttentionModule is a temporal attention module
class TemporalAttentionModule(nn.Module):
    def __init__(self, num_inp_channels: int, num_frames: int, embed_dim: int = 40, num_heads: int = 4) -> None:
        super(TemporalAttentionModule, self).__init__()

        self.num_inp_channels = num_inp_channels
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # create multiheaded attention module
        self.to_q = nn.Linear(num_inp_channels, embed_dim)
        self.to_k = nn.Linear(num_inp_channels, embed_dim)
        self.to_v = nn.Linear(num_inp_channels, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    # forward performs temporal attention on the input (b,t,h,w,c)
    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        grouped_x = rearrange(x, '(b t) c h w -> (b h w) t c', t=self.num_frames)

        # perform self-attention on the grouped_x
        q, k, v = self.to_q(grouped_x), self.to_k(grouped_x), self.to_v(grouped_x)
        attn_out = self.attn(q, k, v)[0]

        # rearrange out to be back into the grouped batch and timestep format
        attn_out = rearrange(attn_out, '(b h w) t c -> (b t) c h w', t=self.num_frames, h=h, w=w)

        return attn_out + x


# ReferenceConditionedAttentionBlock is an attention block which performs spatial and temporal attention
class ReferenceConditionedAttentionBlock(nn.Module):
    def __init__(self, cross_attn: Transformer2DModel, num_frames: int, skip_temporal_attn: bool = False):
        super(ReferenceConditionedAttentionBlock, self).__init__()

        # store configurations and submodules
        self.skip_temporal_attn = skip_temporal_attn
        self.num_frames = num_frames
        self.cross_attn = cross_attn
        self.sam = SpatialAttentionModule()

        # extract channel dimension from provided cross_attn and 
        num_channels = cross_attn.config.in_channels
        embed_dim = cross_attn.config.in_channels
        self.tam = TemporalAttentionModule(num_channels, self.num_frames, embed_dim=embed_dim)

        # store the reference tensor used by this module (this must be updated before the forward pass)
        self.reference_tensor = None

    # update_reference_tensor updates the reference tensor for the module
    def update_reference_tensor(self, reference_tensor: torch.FloatTensor):
        self.reference_tensor = reference_tensor

    # forward performs spatial attention, cross attention, and temporal attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # begin spatial attention
        # expand and concat output with reference embedding
        w = hidden_states.shape[3]
        concat = torch.cat((hidden_states, self.reference_tensor), axis=3)

        # pass concat tensor through spatial attention module along w axis (b,c,h,w)
        out = self.sam(concat)

        # take only the first half of the output concat tensor (b,c,h,w)
        out = out[:, :, :, :w]

        # begin cross attention
        out = self.cross_attn(out, encoder_hidden_states, timestep, added_cond_kwargs, class_labels,
                            cross_attention_kwargs, attention_mask, encoder_attention_mask, return_dict)[0]

        # begin temporal attention
        if self.skip_temporal_attn:
            return (out,)
        
        # pass the data through the temporal attention module
        out = self.tam(out)

        return (out,)


# VideoNet is a unet initialized from stable diffusion used to denoise video frames
class VideoNet(nn.Module):
    def __init__(self, sd_unet: UNet2DConditionModel, num_frames: int = 24, batch_size: int = 2):
        super(VideoNet, self).__init__()
        self.batch_size = batch_size

        # create a deep copy of the sd_unet
        self.unet = copy.deepcopy(sd_unet)

        # maintain a list of all the new ReferenceConditionedResNets and TemporalAttentionBlocks
        self.ref_cond_attn_blocks: List[ReferenceConditionedAttentionBlock] = []

        # replace attention blocks with ReferenceConditionedAttentionBlock
        down_blocks = self.unet.down_blocks
        mid_block = self.unet.mid_block
        up_blocks = self.unet.up_blocks

        for i in range(len(down_blocks)):
            if hasattr(down_blocks[i], "attentions"):
                attentions = down_blocks[i].attentions
                for j in range(len(attentions)):
                    attentions[j] = ReferenceConditionedAttentionBlock(attentions[j], num_frames)
                    self.ref_cond_attn_blocks.append(attentions[j])

        for i in range(len(mid_block.attentions)):
            mid_block.attentions[i] = ReferenceConditionedAttentionBlock(mid_block.attentions[i], num_frames)
            self.ref_cond_attn_blocks.append(mid_block.attentions[i])
        
        for i in range(len(up_blocks)):
            if hasattr(up_blocks[i], "attentions"):
                attentions = up_blocks[i].attentions
                for j in range(len(attentions)):
                    attentions[j] = ReferenceConditionedAttentionBlock(attentions[j], num_frames)
                    self.ref_cond_attn_blocks.append(attentions[j])

    # update_reference_embeddings updates all the reference embeddings in the unet
    def update_reference_embeddings(self, reference_embeddings):
        if len(reference_embeddings) != len(self.ref_cond_attn_blocks):
            print("[!] WARNING - amount of input reference embeddings does not match number of modules in VideoNet")

        for i in range(len(self.ref_cond_attn_blocks)):
            # update the reference conditioned blocks embedding
            self.ref_cond_attn_blocks[i].update_reference_tensor(reference_embeddings[i])

    # update_skip_temporal_attn updates all the skip temporal attention attributes
    def update_skip_temporal_attn(self, skip_temporal_attn):
        for i in range(len(self.ref_cond_attn_blocks)):
            # update the skip_temporal_attn attribute
            self.ref_cond_attn_blocks[i].skip_temporal_attn = skip_temporal_attn

    # forward pass just passes pose + conditioning embeddings to unet and returns activations
    def forward(self, intial_noise, timesteps, reference_embeddings, clip_condition_embeddings, skip_temporal_attn=False):
        # update the reference tensors for the ReferenceConditionedResNet modules
        self.update_reference_embeddings(reference_embeddings)

        # update the skip temporal attention attribute
        self.update_skip_temporal_attn(skip_temporal_attn)

        # forward pass the pose + conditioning embeddings through the unet
        return self.unet(
            intial_noise,
            timesteps,
            encoder_hidden_states=clip_condition_embeddings,
        )[0]

