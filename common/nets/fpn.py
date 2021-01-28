import torch
import torch.nn.functional as F
from torch import nn

from config import cfg
from nets.resnet import resnet50

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        # bottom up layers (resnet50)
        self.bottom_up = resnet50(pretrained=cfg.resnet_pretrained)
        
        # lateral convolutional layer
        self.fpn_lateral2 = nn.Conv2d(256, 256, 1, 1)
        self.fpn_lateral3 = nn.Conv2d(512, 256, 1, 1)
        self.fpn_lateral4 = nn.Conv2d(1024, 256, 1, 1)
        self.fpn_lateral5 = nn.Conv2d(2048, 256, 1, 1)
        
        # output convolutional layer
        self.fpn_output2 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.fpn_output3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.fpn_output4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.fpn_output5 = nn.Conv2d(256, 256, 3, 1, padding=1) 

    def forward(self, x):
        # resnet50 - c2, c3, c4, c5
        x = self.bottom_up(x)

        outputs = {}
        # feature - p5
        feature = self.fpn_lateral5(x[3])
        outputs['p5'] = self.fpn_output5(feature)

        # feature - p4
        top_down_features = F.interpolate(feature, scale_factor=2, mode="nearest")
        lateral_features = self.fpn_lateral4(x[2])
        feature = lateral_features + top_down_features
        outputs['p4'] = self.fpn_output4(feature)

        # feature - p3
        top_down_features = F.interpolate(feature, scale_factor=2, mode="nearest")
        lateral_features = self.fpn_lateral3(x[1])
        feature = lateral_features + top_down_features
        outputs['p3'] = self.fpn_output3(feature)

        # feature - p2
        top_down_features = F.interpolate(feature, scale_factor=2, mode="nearest")
        lateral_features = self.fpn_lateral2(x[0])
        feature = lateral_features + top_down_features
        outputs['p2'] = self.fpn_output2(feature)

        # feature - p6
        outputs['p6'] = F.max_pool2d(outputs['p5'], kernel_size=1, stride=2)

        return outputs

    def freeze(self):
        for p in self.parameters():
            if p.requires_grad:
                p.requires_grad = False