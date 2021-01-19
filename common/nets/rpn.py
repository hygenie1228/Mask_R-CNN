import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import nn
from torchvision import ops

from config import cfg
from utils.func import Box
from utils.visualize import visualize_anchors, visualize_labeled_anchors
from nets.anchor_generator import AnchorGenerator

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        # from config
        self.rpn_features = cfg.rpn_features
        if self.training:
            self.pre_nms_topk = cfg.pre_nms_topk_train
            self.post_nms_topk = cfg.post_nms_topk_train
        else:
            self.pre_nms_topk = cfg.pre_nms_topk_test
            self.post_nms_topk = cfg.post_nms_topk_test
        
        self.nms_threshold = cfg.nms_threshold
        self.anchor_samples = cfg.num_sample

        # anchors per one cell
        self.anchor_generator = AnchorGenerator()

        # layers
        # 3x3 conv layer
        self.conv = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # predict objectness
        if cfg.use_entropy:
            self.objectness_logits = nn.Conv2d(256, 2*len(cfg.anchor_ratios), 1, stride=1)
        else:
            self.objectness_logits = nn.Conv2d(256, len(cfg.anchor_ratios), 1, stride=1)
        # predict proposal coordinate
        self.anchor_deltas = nn.Conv2d(256, 4*len(cfg.anchor_ratios), 1, stride=1)

        self.img = None

    def forward(self, x, img, gt_data):
        self.img = img[0]

        # generate anchors
        total_anchors = self.anchor_generator.get_anchors(x)
        
        # predict
        pred_scores, pred_deltas = self.predict_scores_deltas(x, total_anchors)

        if self.training:
            anchors = torch.cat(total_anchors, dim=0)
            gt_boxes = torch.tensor(gt_data['bboxs']).cuda()

            # labeling & sampling anchors
            anchor_labels, match_gt_idxs = self.anchor_generator.labeling_anchors(anchors, gt_boxes)
            anchor_labels = self.anchor_generator.sampling_anchors(anchor_labels)
            
            # loss function
            cls_loss, loc_loss = self.loss(anchors, gt_boxes, pred_scores, pred_deltas, anchor_labels, match_gt_idxs)
        
        self.get_proposals(total_anchors, pred_scores, pred_deltas, img)

        
        # visualize anchors
        if cfg.visualize & self.training:
            #idxs = torch.randperm(len(anchors))[:1000]
            #sample_anchors = anchors[idxs]
            #visualize_anchors(img[0], sample_anchors, './outputs/anchor_image.jpg')
            #visualize_anchors(img[0], total_anchors[4][200:220], './outputs/anchor_image.jpg')

            idxs = torch.where(anchor_labels == 1)[0]
            pos_anchors = anchors[idxs]
            idxs = torch.where(anchor_labels == 0)[0]
            neg_anchors = anchors[idxs]
            visualize_labeled_anchors(self.img, gt_data['bboxs'], pos_anchors, neg_anchors, './outputs/labeled_anchor_image.jpg')

        return cls_loss, loc_loss

    def predict_scores_deltas(self, x, anchors):
        pred_scores = []
        pred_deltas = []
        for lvl in self.rpn_features:
            scores, deltas = self.rpn_forward(x[lvl])
            pred_scores.append(scores.squeeze(0))
            pred_deltas.append(deltas.squeeze(0))

        # check dimension
        for anchors, scores in zip(anchors, pred_scores):
            if anchors.shape[0] != scores.shape[0]:
                print("Not same anchor counts !!!")
                raise ValueError

        return pred_scores, pred_deltas

    def rpn_forward(self, x):
        x = F.relu(self.conv(x))
        pred_scores = self.objectness_logits(x)     # [bs, 6, h, w]
        pred_deltas = self.anchor_deltas(x)         # [bs, 12, h, w]     

        bs, _, h, w = pred_scores.shape
        
        if cfg.use_entropy:
            pred_scores = pred_scores.reshape(bs, 2, -1)   # [bs, 2, 3*h*w]
            pred_scores = F.softmax(pred_scores, 1)        # [bs, 2, 3*h*w]
        else:
            pred_scores = pred_scores.reshape(bs, 1, -1)   # [bs, 2, 3*h*w]

        pred_deltas = pred_deltas.reshape(bs, 4, -1)   # [bs, 4, 3*h*w]          

        pred_scores = pred_scores.permute(0, 2, 1)  # [bs, 3*h*w, 2]
        pred_deltas = pred_deltas.permute(0, 2, 1)  # [bs, 3*h*w, 4]

        return pred_scores, pred_deltas

    def loss(self, anchors, gt_boxes, pred_scores, pred_deltas, anchor_labels, match_gt_idxs):
        # matching
        pred_scores = torch.cat(pred_scores, dim=0)
        pred_deltas = torch.cat(pred_deltas, dim=0)

        # delta
        idxs = torch.where(anchor_labels == 1)[0]
        #print("!!!")
        #print(idxs)
        pos_anchors = anchors[idxs]                 # anchor
        match_gt_idxs = match_gt_idxs[idxs]         
        match_gt_boxes = gt_boxes[match_gt_idxs]    # gt box
        
        gt_delta = Box.pos_to_delta(match_gt_boxes, pos_anchors)    #target
        pos_deltas = pred_deltas[idxs]              # pred
        
        # scores
        idxs = torch.where(anchor_labels >= 0)[0]
        gt_labels = anchor_labels[idxs]             # target
        mask_scores = pred_scores[idxs]             # pred

        # debugging...
        d_scores, d_idx = pred_scores[:, 1].sort(descending=True)
        d_scores, d_topk_idx = d_scores[:10], d_idx[:10]
        d_topk_idx, _ = d_topk_idx.sort()
        #print(d_topk_idx)

        # debugging...
        scores, idx = pred_scores[:, 1].sort(descending=True)
        scores, topk_idx = scores[:20], idx[:20]

        d_delta = pred_deltas[topk_idx]
        d_anchors = anchors[topk_idx]

        pos_proposal = Box.delta_to_pos(d_anchors, d_delta)
        visualize_labeled_anchors(self.img, match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_score_image.jpg')

        if cfg.visualize:
            pos_proposal = Box.delta_to_pos(pos_anchors, pos_deltas)
            visualize_labeled_anchors(self.img, match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_anchor_image.jpg')

            pos_proposal = Box.delta_to_pos(d_anchors, d_delta)
            visualize_labeled_anchors(self.img, match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_score_image.jpg')

        #cls_loss = F.cross_entropy(pred_scores, anchor_labels.long(), ignore_index = -1)
        cls_loss = F.cross_entropy(mask_scores, gt_labels.long(), ignore_index = -1)
        cls_loss = cls_loss / self.anchor_samples

        loc_loss = self.smooth_l1_loss(pos_deltas, gt_delta, beta=cfg.smooth_l1_beta)
        loc_loss = loc_loss / self.anchor_samples

        return cls_loss, loc_loss

    def smooth_l1_loss(self, bbox_pred, bbox_targets, beta=1.0):
        box_diff = bbox_pred - bbox_targets
        abs_in_box_diff = torch.abs(box_diff)
        smoothL1_sign = (abs_in_box_diff < beta).detach().float()
        loss_box = smoothL1_sign * 0.5 * torch.pow(box_diff, 2) / beta + \
                    (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
        N = loss_box.size(0)  # batch size
        loss_box = loss_box.view(-1).sum(0) / N
        return loss_box

    def get_proposals(self, total_anchors, pred_scores, pred_deltas, img):
        img_size = (img.shape[2], img.shape[3])

        pred_proposals = []
        proposal_scores = []

        for scores, deltas, anchors in zip(pred_scores, pred_deltas, total_anchors):
            scores = scores[:,1]

            # pre nms topk
            K = min(self.pre_nms_topk, len(scores))
            scores, idx = scores.sort(descending=True)
            scores, topk_idx = scores[:K], idx[:K]

            # get proposals
            topk_deltas = deltas[topk_idx]
            topk_anchors = anchors[topk_idx]
            proposals = Box.delta_to_pos(topk_anchors, topk_deltas)

            pred_proposals.append(proposals)
            proposal_scores.append(scores)
        
        pred_proposals = torch.cat(pred_proposals, dim=0)
        proposal_scores = torch.cat(proposal_scores, dim=0)

        # valid check
        proposal_scores, pred_proposals = Box.box_valid_check(proposal_scores, pred_proposals, img_size)

        # post nms topk
        topk_idx = ops.nms(pred_proposals, proposal_scores, self.nms_threshold)

        K = min(self.post_nms_topk, len(topk_idx))
        topk_idx = topk_idx[:K]
        
        proposal_scores = proposal_scores[topk_idx]
        pred_proposals = pred_proposals[topk_idx]
        #print(pred_proposals.shape)

        #print(proposal_scores[:100])
            
        if cfg.visualize:
            visualize_anchors(self.img, pred_proposals[:100], './outputs/proposal_image.jpg')
        

        #proposal_scores, idx = proposal_scores.sort(descending=True)
        #topk_idx = idx[: 200]
        
        
        