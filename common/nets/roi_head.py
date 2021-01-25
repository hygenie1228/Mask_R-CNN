import torch
from torch import nn

from config import cfg
from utils.func import Box
from utils.visualize import visualize_anchors, visualize_labeled_anchors

class ROIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = cfg.roi_threshold
        self.num_sample = cfg.roi_num_sample
        self.positive_ratio = cfg.roi_positive_ratio
        self.num_labels = cfg.num_labels
        
        self.img = None

    def forward(self, x, proposals, images, gt_datas):
        # for debugging
        self.img = images[0]

        if self.training:
            gt_labels = [torch.tensor(gt_data['category_id']).cuda() for gt_data in gt_datas]
            gt_boxes = [torch.tensor(gt_data['bboxs']).cuda() for gt_data in gt_datas]

            # add gt_box to proposals
            proposals = self.add_gt_boxes(proposals, gt_boxes)
            # labeling
            proposal_labels, match_gt_boxes = self.labeling_proposals(proposals, gt_labels, gt_boxes)
            proposal_labels = self.sampling_proposals(proposal_labels)

        if cfg.visualize & self.training:    
            idxs = torch.where(proposal_labels[0] >= 1)[0]
            pos_proposals = proposals[0][idxs]
            idxs = torch.where(proposal_labels[0] == 0)[0]
            neg_proposals = proposals[0][idxs]
            visualize_labeled_anchors(self.img, gt_datas[0]['bboxs'], pos_proposals, neg_proposals, './outputs/labeled_proposal_image.jpg')

    def labeling_proposals(self, proposals, gt_labels, gt_boxes):
        # proposals : [bs, N, 4], format : x1y1x2y2
        # gt_boxes : [bs, M, 4], format : x1y1x2y2
        
        labels = []
        match_gt_boxes = []
        
        for i, (proposal, gt_label, gt_box) in enumerate(zip(proposals, gt_labels, gt_boxes)):
            label = torch.empty(len(proposal),).cuda().long().fill_(0)
            match_gt_box = torch.zeros(len(proposal), 4).cuda()

            # get iou_matrix : [N, M]
            iou_matrix  = Box.calculate_iou_matrix(proposal, gt_box)

            # labeling
            max_ious, match_gt_idxs = torch.max(iou_matrix, dim=1)
            max_ious_idxs = torch.where(max_ious > self.threshold)[0]
            match_gt_idxs = match_gt_idxs[max_ious_idxs]

            # labels & coressponding gt boxes
            label[max_ious_idxs] = gt_label[match_gt_idxs]
            match_gt_box[max_ious_idxs] = gt_box[match_gt_idxs]

            labels.append(label)
            match_gt_boxes.append(match_gt_box)

        # labels : [bs, N] 
        # match_gt_boxes : [bs, N, 4]
        return labels, match_gt_boxes

    def sampling_proposals(self, input_labels):
        # input_labels : [bs, N]
        
        sampling_labels = []
        for labels in input_labels:
            sampling_label = torch.empty(len(labels),).cuda().long().fill_(-1)
            
            pos_index = torch.where(labels >= 1)[0]
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
            
            sampling_label[pos_index] = labels[pos_index]
            sampling_label[neg_index] = 0

            sampling_labels.append(sampling_label)

        return sampling_labels

    def add_gt_boxes(self, proposals, gt_boxes):
        new_proposals = []
        for proposal, gt_box in zip(proposals, gt_boxes):
            new_proposals.append(torch.cat([proposal, gt_box], dim=0))
        
        return new_proposals