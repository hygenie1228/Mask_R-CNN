import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from config import cfg
from nets.fpn import FPN
from nets.rpn import RPN
from nets.roi_align import ROIAlign
from nets.roi_head import ROIHead
from utils.visualize import visualize_input_image, visualize_labeled_anchors

class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = FPN()

        # use pretrained FPN ?
        #if cfg.fpn_pretrained:
        #    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)

        self.rpn = RPN()

        # roi head
        self.roi_align = ROIAlign()
        self.roi_head = ROIHead()
        
    def forward(self, data):
        '''
        images = [d['image'].cuda().unsqueeze(0) for d in data]
        gt_datas = [d['gt_data'] for d in data]

        for image, gt_data in zip(images, gt_datas):
            features = self.fpn(image)
            cls_loss, loc_loss = self.rpn(features, image, gt_data)
        '''

        images = [d['image'] for d in data]
        gt_datas = [d['gt_data'] for d in data]
        
        images, gt_datas = self.recompose_batch(images, gt_datas)

        #for image, gt_data in zip(images, gt_datas):
        features = self.fpn(images)
        cls_loss, loc_loss, proposals = self.rpn(features, images, gt_datas)

        self.roi_head(features, proposals, images, gt_datas)

        # visualize input image
        if cfg.visualize & self.training :  
            idx = 0
            img = images[idx]
            gt_data = gt_datas[idx]
            img = img.cpu().numpy().transpose(1, 2, 0)
            visualize_input_image(img, gt_data, './outputs/input_image.jpg')

            #gt_img = data[idx]['raw_image']
            #raw_gt_data = data[idx]['raw_gt_data']
            #visualize_input_image(gt_img, raw_gt_data, './outputs/gt_image.jpg')

        return cls_loss, loc_loss, 22

    def recompose_batch(self, images, gt_datas):
        _, h, w = images[0].shape

        # remove not match ratio
        if w/h > 1:
            w_long = True
        else:
            w_long = False

        process_images = [images[0]]
        process_gt_datas = [gt_datas[0]]
        max_w = w
        max_h = h
        for image, gt_data in zip(images[1:], gt_datas[1:]):
            _, h, w = image.shape
            if (w_long & (w/h > 1)) or (~w_long & (w/h < 1)):
                process_images.append(image)
                process_gt_datas.append(gt_data)
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h

        # recompose
        padded_imgs = torch.zeros((len(process_images), 3, max_h, max_w)).cuda()
        for i, image in enumerate(process_images):
            _, h, w = image.shape
            padded_imgs[i, :, :h, :w] = image

        return padded_imgs, process_gt_datas



