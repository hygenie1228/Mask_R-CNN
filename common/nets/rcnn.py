import torch
from torch import nn

from config import cfg
from nets.fpn import FPN
from nets.rpn import RPN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = FPN()

        # use pretrained FPN ?
        #if cfg.fpn_pretrained:
        #    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)

        self.rpn = RPN()
        
    def forward(self, data):
        images = [d['image'].cuda().unsqueeze(0) for d in data]
        gt_datas = [d['gt_data'] for d in data]

        '''
        for d in data:
            d['gt_data']['img_size'] = d['img_size']
            gt_datas.append(d['gt_data'])
        '''
        for image, gt_data in zip(images, gt_datas):
            features = self.fpn(image)
            cls_loss = self.rpn(features, image, gt_data)

            break

        return cls_loss