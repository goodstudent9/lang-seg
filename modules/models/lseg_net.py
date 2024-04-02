import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os
# from dinov2_encoder import dinov2_encoder, projection_net
from transformers import Dinov2Model, Dinov2PreTrainedModel, DPTModel, Dinov2Config, DPTConfig, DPTForDepthEstimation
from transformers.modeling_outputs import SemanticSegmenterOutput

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        # hooks = {
        #     "clip_vitl16_384": [5, 11, 17, 23],
        #     "clipRN50x16_vitl16_384": [5, 11, 17, 23],
        #     "clip_vitb32_384": [2, 5, 8, 11],
        # }

        # # # Instantiate backbone and reassemble blocks
        # self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
        #     backbone,
        #     features,
        #     groups=1,
        #     expand=False,
        #     exportable=False,
        #     hooks=hooks[backbone],
        #     use_readout=readout,
        # )
        self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        # self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        # self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        # self.arch_option = kwargs["arch_option"]
        # if self.arch_option == 1:
        #     self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
        #     self.block_depth = kwargs['block_depth']
        # elif self.arch_option == 2:
        #     self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
        #     self.block_depth = kwargs['block_depth']

        # self.scratch.output_conv = head

        self.text = clip.tokenize(self.labels)  
        #自己的代码
        # self.dinov2 = dinov2_encoder.from_pretrained("facebook/dinov2-base")
        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-large", out_features=["stage2", "stage5", "stage8", "stage11"], reshape_hidden_states=False)

        # config = DPTConfig(backbone_config=backbone_config)
        config = DPTConfig(backbone_config=backbone_config, image_size=480)
        self.dinov2  = DPTForDepthEstimation(config=config)
        self.up_d = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
        self.up_d = nn.Conv2d(features, self.out_c, kernel_size=1)
        # self.head_dinov2 = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU()
        # )
        # self.text_projection = nn.Linear(512, 768, bias=False)
        # self.project = projection_net()
        #结束

    def forward(self, x, labelset=''):
        if labelset == '':
            text = self.text
        else:
            text = clip.tokenize(labelset)    
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        #TODO 自己写的dinov2的embedding
        # with torch.no_grad():  
        patch_embeddings = self.dinov2(x, output_hidden_states=True, return_dict=True)
        dinov2_hidden_states = patch_embeddings.hidden_states #4 256 272 272
        # get the patch emeddings - so we exclude the CLS token
        #自己的代码结束
        last_fused_feature = dinov2_hidden_states[-1]

        # layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x) #图片的大小是1,3,480,480
        # # layer_1, layer_2, layer_3, layer_4 = forward_dinov2(dinov2_hidden_states)


        # layer_1_rn = self.scratch.layer1_rn(layer_1) # 4 256 120 120
        # layer_2_rn = self.scratch.layer2_rn(layer_2) # 4 512 60 60
        # layer_3_rn = self.scratch.layer3_rn(layer_3) # 4 1024 30 30
        # layer_4_rn = self.scratch.layer4_rn(layer_4) # 4 2048 15 15

        # path_4 = self.scratch.refinenet4(layer_4_rn)
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text).float()
        # text_features = self.text_projection(text_features)

        # image_features = self.scratch.head1(path_1) #4 256 240 240
        image_features = self.up_d(last_fused_feature)
        # image_features = self.scratch.head1(last_fused_feature) #4 256 272 272
        #TODO 这里需要进行patch的恢复，同时需要进行linear从768到512维度的变换
        # patch_num = int(math.sqrt(patch_embeddings.shape[1]))

        # image_features = patch_embeddings.reshape(-1, patch_num ,patch_num ,768) #1 34 34 768
        # image_features = image_features.permute(0, 3, 1, 2)
        # #先进行扩充，然后再去降维
        # image_features = torch.nn.functional.interpolate(image_features, size=(240, 240), mode='bicubic', align_corners=False) #1 240 240 768
        # image_features = self.project(image_features)
        #自己的代码结束

        imshape = image_features.shape #这里的imagefeature的shape是 1 512 240 240
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.logit_scale * image_features @ text_features.t() # 512

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        # if self.arch_option in [1, 2]:
        #     for _ in range(self.block_depth - 1):
        #         out = self.scratch.head_block(out)
        #     out = self.scratch.head_block(out, False)
        #这里进行了插值函数，使其恢复到原始的图像大小
        # out = self.scratch.output_conv(out)
        out =  F.interpolate(out, size=(480, 480), mode='bilinear', align_corners=False)
        return out #1 150 480 480


class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


    
        
    