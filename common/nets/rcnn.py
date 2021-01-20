import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from config import cfg
from nets.fpn import FPN
from nets.rpn import RPN
from utils.visualize import visualize_input_image, visualize_labeled_anchors

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


        for image, gt_data in zip(images, gt_datas):
            features = self.fpn(image)
            cls_loss = self.rpn(features, image, gt_data)

        # visualize input image
        if cfg.visualize & self.training :  
            img = images[0][0]
            img = img.cpu().numpy().transpose(1, 2, 0)
            visualize_input_image(img, gt_data, './outputs/input_image.jpg')

            gt_img = data[0]['raw_image']
            raw_gt_data = data[0]['raw_gt_data']
            visualize_input_image(gt_img, raw_gt_data, './outputs/gt_image.jpg')

        return cls_loss