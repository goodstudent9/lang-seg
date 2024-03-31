from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import torch
import torch.nn as nn


class dinov2_encoder(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    # config.hidden_size=512
    self.dinov2 = Dinov2Model(config)

  def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
    outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    # get the patch embeddings - so we exclude the CLS token
    patch_embeddings = outputs.last_hidden_state[:,1:,:]


    return patch_embeddings


class projection_net(nn.Module):
    def __init__(self):
        super(projection_net, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0)
        # 使用BatchNorm进行归一化，提升网络性能
        self.bn1 = nn.BatchNorm2d(512)
        # 定义激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入的张量维度为 1x34x34x768
        # 进行卷积操作
        x = self.conv1(x)
        # 进行BatchNorm归一化
        # x = self.bn1(x)
        # # 使用ReLU激活函数
        # x = self.relu(x)
        return x
    
# class LinerProject(nn.Module):
#    def __init__(self, *args, **kwargs):
#       super(LinerProject, self).__init__()
#       self.linear = nn.Linear(768, 512)
#       self.up_sample = nn.Linear() #

