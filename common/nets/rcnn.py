import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from config import cfg
from nets.fpn import FPN
from nets.rpn import RPN
from nets.roi_head import ROIHead
from utils.func import Box

class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Networks
        self.fpn = FPN()
        self.rpn = RPN()
        self.roi_head = ROIHead()
        
    def forward(self, datas):
        # compose batch
        images, gt_datas = self.pre_processing(datas)
        
        # Feed images
        features = self.fpn(images)
        proposal_loss, proposals = self.rpn(features, images, gt_datas)
        detection_loss, results = self.roi_head(features, proposals, images, gt_datas)
        #detection_loss = (torch.tensor([0.0]).cuda(), torch.tensor([0.0]).cuda())
        
        if self.training:
            return proposal_loss, detection_loss
        else:
            results = self.post_processing(datas, results)
            return results        

    def pre_processing(self, data):
        if ~self.training & (len(data) > 1):
            raise Exception('[ERROR] Only support batch size = 1 !!!')
            
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

    def post_processing(self, datas, results):
        final_results = []

        for data, result in zip(datas, results):
            image_id = data['img_id']
            ratio_h = data['raw_img_size'][0] / data['img_size'][0]
            ratio_w = data['raw_img_size'][1] / data['img_size'][1]
            for result_per_label in result:
                category_id = result_per_label['label']
                result_per_label['bbox'][:,(0,2)] = result_per_label['bbox'][:,(0,2)] * ratio_w
                result_per_label['bbox'][:,(1,3)] = result_per_label['bbox'][:,(1,3)] * ratio_h

                for score, bbox in zip(result_per_label['score'], result_per_label['bbox']):
                    final_results.append({
                        'image_id' : int(image_id),
                        'category_id' : int(category_id),
                        'bbox' : Box.xyxy_to_xywh(bbox.tolist()),
                        'score' : round(score.tolist(),3)
                    })

        return final_results

