import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from config import cfg
from nets.fpn import FPN
from nets.rpn import RPN
from nets.roi_head import ROIHead
from utils.visualize import visualize_input_image

class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Networks
        self.fpn = FPN()
        self.rpn = RPN()
        self.roi_head = ROIHead()

        #if cfg.fpn_pretrained:
        #    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        
    def forward(self, data):
        # compose batch
        images, gt_datas = self.recompose_batch(data)
        
        # Feed images
        features = self.fpn(images)
        proposal_loss, proposals = self.rpn(features, images, gt_datas)
        detection_loss, results = self.roi_head(features, proposals, images, gt_datas)
        
        # visualize input image
        if cfg.visualize & self.training :
            img = images[0]
            gt_data = gt_datas[0]
            visualize_input_image(images[0], gt_datas[0]['bboxs'], './outputs/input_image.jpg')

            '''
            gt_img = data[0]['raw_image']
            raw_gt_data = data[0]['raw_gt_data']
            visualize_input_image(gt_img, raw_gt_data, './outputs/gt_image.jpg')
            '''

        return proposal_loss, detection_loss

    def recompose_batch(self, data):
        images = [d['image'] for d in data]
        gt_datas = [d['gt_data'] for d in data]

        _, h, w = images[0].shape

        # remove not match image ratio
        w_long = (w/h > 1)

        process_images = [images[0]]
        process_gt_datas = [gt_datas[0]]
        max_h, max_w = h, w
        
        for image, gt_data in zip(images[1:], gt_datas[1:]):
            _, h, w = image.shape
            if (w_long & (w/h > 1)) or (~w_long & (w/h < 1)):
                process_images.append(image)
                process_gt_datas.append(gt_data)
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h

        # recompose batch
        padded_imgs = torch.zeros((len(process_images), 3, max_h, max_w)).cuda()
        for i, image in enumerate(process_images):
            _, h, w = image.shape
            padded_imgs[i, :, :h, :w] = image

        return padded_imgs, process_gt_datas

