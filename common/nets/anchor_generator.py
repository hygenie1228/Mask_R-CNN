import torch
import math

from config import cfg
from utils.func import Box

class AnchorGenerator:
    def __init__(self):
        self.rpn_features = cfg.rpn_features
        self.anchor_scales = cfg.anchor_scales
        self.anchor_ratios = cfg.anchor_ratios
        self.anchor_strides = cfg.anchor_strides

        self.positive_anchor_threshold = cfg.positive_anchor_threshold
        self.negative_anchor_threshold = cfg.negative_anchor_threshold
        self.num_sample = cfg.num_sample
        self.positive_ratio = cfg.positive_ratio

        self.basic_anchors = self.generate_basic_anchors() 

    def get_anchors(self, x):
        total_anchors = []
        for i, lvl in enumerate(self.rpn_features):
            feature_size = (x[lvl].shape[2], x[lvl].shape[3])
            stride = self.anchor_strides[i]
            basic_anchors = self.basic_anchors[i]
            basic_anchors = basic_anchors.repeat(feature_size[0], feature_size[1], 1, 1)
            
            anchors = []

            y_range = torch.arange(0, feature_size[1]).repeat(feature_size[0], 1).reshape(feature_size[0], feature_size[1], 1)
            x_range = torch.arange(0, feature_size[0]).repeat(1, feature_size[1]).reshape(feature_size[1], feature_size[0], 1).permute(1, 0, 2)

            anchors = torch.cat([y_range, x_range, y_range, x_range], dim=2) * stride + stride / 2
            anchors = anchors.repeat(3, 1, 1, 1).permute(1, 2, 0 ,3)
            anchors = (anchors + basic_anchors).reshape(-1, 4)

            anchors = anchors.cuda()
            total_anchors.append(anchors)
            
        return total_anchors

    def generate_basic_anchors(self):
        anchors = []
        for scale in self.anchor_scales:
            anchors_one_scale = []
            area = scale ** 2.0     # area = scale ^ 2
            for ratio in self.anchor_ratios:
                w = math.sqrt(area / ratio) 
                h = ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors_one_scale.append([x0, y0, x1, y1])
            anchors.append(anchors_one_scale)
        return torch.tensor(anchors)

    def labeling_anchors(self, anchors, gt_boxes):
        # anchors : [N, 4], format : x1y1x2y2
        # gt_boxes : [bs, M, 4], format : x1y1x2y2
        
        labels = torch.empty(len(gt_boxes), len(anchors)).cuda().fill_(-1)
        match_gt_boxes = torch.empty(len(gt_boxes), len(anchors), 4).cuda().fill_(-1)
        
        for i, gt_box in enumerate(gt_boxes):
            # get iou_matrix : [N, M]
            iou_matrix  = Box.calculate_iou_matrix(anchors, gt_box)

            # anchor labels
            # 1 : positive anchor
            # 0 : negative anchor
            # -1 : don't care
            max_ious, match_gt_idxs = torch.max(iou_matrix, dim=1)
            gt_max_ious, _ = torch.max(iou_matrix, dim=0)
            gt_argmax_ious = torch.where(iou_matrix == gt_max_ious)[0]

            # assign negative anchor
            labels[i, max_ious < self.negative_anchor_threshold] = 0

            # assign positive anchor
            labels[i, max_ious >= self.positive_anchor_threshold] = 1
            labels[i, gt_argmax_ious] = 1
            
            # match gt boxes
            match_gt_boxes[i] = gt_box[match_gt_idxs.long()]
        
        return labels, match_gt_boxes

    def labeling_anchors_2(self, anchors, gt_boxes):
        # anchors : [N, 4], format : x1y1x2y2
        # gt_boxes : [bs, M, 4], format : x1y1x2y2
        
        bs = len(gt_boxes)
        gt_boxes = gt_boxes[0]
        
        new_labels = torch.empty(bs, len(anchors)).cuda().fill_(-1)
        new_match_gt_idxs = torch.empty(bs, len(anchors)).cuda().fill_(-1)
        
        # get iou_matrix : [N, M]
        iou_matrix  = Box.calculate_iou_matrix(anchors, gt_boxes)

        # anchor labels
        # 1 : positive anchor
        # 0 : negative anchor
        # -1 : don't care
        labels = torch.empty(len(anchors),).cuda().fill_(-1)
        max_ious, match_gt_idxs = torch.max(iou_matrix, dim=1)
        gt_max_ious, _ = torch.max(iou_matrix, dim=0)
        gt_argmax_ious = torch.where(iou_matrix == gt_max_ious)[0]

        # assign negative anchor
        labels[max_ious < self.negative_anchor_threshold] = 0

        # assign positive anchor
        labels[max_ious >= self.positive_anchor_threshold] = 1
        labels[gt_argmax_ious] = 1

        print(match_gt_idxs.shape)
        return labels, match_gt_idxs
    
    def sampling_anchors(self, input_labels):
        # input_labels : [bs, N]
        sampling_labels = torch.empty(input_labels.shape).cuda().fill_(-1)

        for i, labels in enumerate(input_labels):
            pos_index = torch.where(labels == 1)[0]
            neg_index = torch.where(labels == 0)[0]
            sampling_pos_num = int(self.num_sample * self.positive_ratio)
            sampling_neg_num = int(self.num_sample * (1 - self.positive_ratio))

            # calculate sampling number
            if pos_index.numel() < sampling_pos_num:
                sampling_pos_num = pos_index.numel()
                sampling_neg_num = int(pos_index.numel() * (1 - self.positive_ratio) /  self.positive_ratio)

            if neg_index.numel() < sampling_neg_num:
                sampling_neg_num = neg_index.numel()
                sampling_pos_num = int(neg_index.numel() * self.positive_ratio / (1 - self.positive_ratio)) 
                
            rand_idx = torch.randperm(pos_index.numel())[:sampling_pos_num]
            pos_index = pos_index[rand_idx]
            rand_idx = torch.randperm(neg_index.numel())[:sampling_neg_num]
            neg_index = neg_index[rand_idx]

            # reassign label
            sampling_labels[i, pos_index] = 1
            sampling_labels[i, neg_index] = 0

        return sampling_labels

    def sampling_anchors_3(self, labels):
        # labels : [bs, N]
        sampling_labels = torch.empty(labels.shape).cuda().fill_(-1)

        pos_index = torch.where(labels == 1)[0]
        neg_index = torch.where(labels == 0)[0]
        sampling_pos_num = int(self.num_sample * self.positive_ratio)
        sampling_neg_num = int(self.num_sample * (1 - self.positive_ratio))

        # calculate sampling number
        if pos_index.numel() < sampling_pos_num:
            sampling_pos_num = pos_index.numel()
            sampling_neg_num = int(pos_index.numel() * (1 - self.positive_ratio) /  self.positive_ratio)

        if neg_index.numel() < sampling_neg_num:
            sampling_neg_num = neg_index.numel()
            sampling_pos_num = int(neg_index.numel() * self.positive_ratio / (1 - self.positive_ratio)) 
            
        rand_idx = torch.randperm(pos_index.numel())[:sampling_pos_num]
        pos_index = pos_index[rand_idx]
        rand_idx = torch.randperm(neg_index.numel())[:sampling_neg_num]
        neg_index = neg_index[rand_idx]

        # reassign label
        labels = torch.empty(labels.numel(),).cuda().fill_(-1)
        labels[pos_index] = 1
        labels[neg_index] = 0

        return labels

    def sampling_anchors_2(self, labels):
        pos_index = torch.where(labels == 1)[0]
        neg_index = torch.where(labels == 0)[0]

        sampling_pos_num = min(pos_index.numel(), int(self.num_sample * self.positive_ratio))
        sampling_neg_num = min(neg_index.numel(), int(self.num_sample * (1 - self.positive_ratio)))

        rand_idx = torch.randperm(pos_index.numel())[:sampling_pos_num]
        pos_index = pos_index[rand_idx]
        rand_idx = torch.randperm(neg_index.numel())[:sampling_neg_num]
        neg_index = neg_index[rand_idx]

        # reassign label
        labels = torch.empty(len(labels),).cuda().fill_(-1)
        labels[pos_index] = 1
        labels[neg_index] = 0

        return labels