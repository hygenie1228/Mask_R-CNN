import torch
from torch import nn
import torch.nn.functional as F
from torchvision import ops

from config import cfg
from utils.func import Box
from utils.losses import smooth_l1_loss
from utils.visualize import visualize_box, visualize_labeled_box
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
        
        self.nms_threshold = cfg.rpn_nms_threshold
        self.anchor_samples = cfg.anchor_num_sample
        self.num_anchor_ratios = len(cfg.anchor_ratios)

        # anchor generator module
        self.anchor_generator = AnchorGenerator()

        # layers
        # 3x3 conv layer
        self.conv = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # predict objectness
        self.objectness_logits = nn.Conv2d(256, self.num_anchor_ratios, 1, stride=1)
        # predict proposal coordinate
        self.anchor_deltas = nn.Conv2d(256, 4*self.num_anchor_ratios, 1, stride=1)

    def forward(self, features, images, gt_datas):
        # for debugging
        self.img = images[0]

        # generate anchors
        total_anchors = self.anchor_generator.get_anchors(features)
        
        # predict scores, deltas
        pred_scores, pred_deltas = self.predict_scores_deltas(features)

        if self.training:
            # to device
            anchors = torch.cat(total_anchors, dim=0)
            gt_boxes = [torch.tensor(gt_data['bboxs']).cuda() for gt_data in gt_datas]

            # labeling & sampling anchors
            anchor_labels, match_gt_boxes = self.anchor_generator.labeling_anchors(anchors, gt_boxes)
            anchor_labels = self.anchor_generator.sampling_anchors(anchor_labels)
            
            # calculate losses
            proposal_loss = self.loss(anchors, match_gt_boxes, pred_scores, pred_deltas, anchor_labels)
        else:
            proposal_loss = (0.0, 0.0)

        # get top proposals
        proposals = self.get_proposals(total_anchors, pred_scores, pred_deltas, images)
        
        # visualize anchors
        if cfg.visualize & self.training:
            idxs = torch.where(anchor_labels[0] == 1)[0]
            pos_anchors = anchors[idxs]
            idxs = torch.where(anchor_labels[0] == 0)[0]
            neg_anchors = anchors[idxs]
            visualize_labeled_box(self.img, gt_datas[0]['bboxs'], pos_anchors, neg_anchors, './outputs/labeled_anchor_image.jpg')
        
        return proposal_loss, proposals

    def predict_scores_deltas(self, x):
        pred_scores = []
        pred_deltas = []
        for lvl in self.rpn_features:
            scores, deltas = self.rpn_forward(x[lvl])
            pred_scores.append(scores)
            pred_deltas.append(deltas)

        # reshape scores & deltas
        # scores : [5, bs, N]
        # deltas : [5, bs, N, 4]
        N, bs = len(pred_scores), pred_scores[0].shape[0]
        reshape_scores = []
        reshape_deltas = []

        for i in range(bs):
            scores = []
            deltas = []
            for j in range(N):
                scores.append(pred_scores[j][i])
                deltas.append(pred_deltas[j][i])
            reshape_scores.append(scores)
            reshape_deltas.append(deltas)

        # scores : [bs, 5, N]
        # deltas : [bs, 5, N, 4]
        return reshape_scores, reshape_deltas

    def rpn_forward(self, x):
        x = F.relu(self.conv(x))                                                    # [bs, 256, h, w]
        pred_scores = self.objectness_logits(x)                                     # [bs, 3, h, w]
        pred_deltas = self.anchor_deltas(x)                                         # [bs, 12, h, w]     
        
        # scores
        bs, _, H, W = pred_scores.shape
        pred_scores = pred_scores.permute(0, 2, 3, 1)                               # [bs, H, W, 3]
        pred_scores = pred_scores.reshape(bs, H*W*self.num_anchor_ratios)           # [bs, H X W X 3]

        # deltas
        bs, _, H, W = pred_deltas.shape
        pred_deltas = pred_deltas.reshape(bs, self.num_anchor_ratios, 4, H, W)      # [bs, 3, 4, H, W]
        pred_deltas = pred_deltas.permute(0, 3, 4, 1, 2)                            # [bs, H, W, 3, 4]
        pred_deltas = pred_deltas.reshape(bs, H*W*self.num_anchor_ratios, 4)        # [bs, H X W X 3, 4]

        return pred_scores, pred_deltas

    def loss(self, anchors, match_gt_boxes, pred_scores, pred_deltas, anchor_labels):
        '''
            anchors :       [N, 4]
            match_gt_boxes: [bs, N, 4]
            pred_scores :   [bs, 5, n]
            pred_deltas :   [bs, 5, n, 4]
            anchor_labels : [bs, N]
        '''

        batch_size = len(anchor_labels)

        reshape_anchors = anchors.repeat(batch_size, 1)
        reshape_match_gts = match_gt_boxes.view(-1 ,4)

        reshape_scores = []
        reshape_deltas = []
        for scores, deltas in zip(pred_scores, pred_deltas):
            reshape_scores.append(torch.cat(scores, dim=0))
            reshape_deltas.append(torch.cat(deltas, dim=0))

        reshape_scores = torch.cat(reshape_scores, dim=0)
        reshape_deltas = torch.cat(reshape_deltas, dim=0)

        reshape_labels = anchor_labels.view(-1)

        '''
            reshape_scores :         [bs X N]
            reshape_labels :         [bs X N]
            reshape_anchors  :       [bs X N, 4]
            reshape_match_gts :      [bs X N, 4]
            reshape_deltas :         [bs X N, 4]
        '''

        # scores matching
        idxs = torch.where(reshape_labels >= 0)[0]
        gt_labels = reshape_labels[idxs]                             # target
        mask_scores = reshape_scores[idxs]                           # pred

        # deltas matching
        idxs = torch.where(reshape_labels == 1)[0]
        pos_anchors = reshape_anchors[idxs]                          # anchor
        pos_gt_boxes = reshape_match_gts[idxs]                       # gt boxes
        
        gt_delta = Box.pos_to_delta(pos_gt_boxes, pos_anchors)       # target
        pos_deltas = reshape_deltas[idxs]                            # pred
        
        # score loss function
        cls_loss = F.binary_cross_entropy_with_logits(mask_scores, gt_labels.to(torch.float32), reduction="sum")

        # delta loss function
        loc_loss = smooth_l1_loss(pos_deltas, gt_delta, beta=cfg.smooth_l1_beta)

        # normalizer
        cls_loss = cls_loss / (batch_size * self.anchor_samples)
        loc_loss = loc_loss / (batch_size * self.anchor_samples)

        #cls_loss = cls_loss / (batch_size * len(gt_labels))
        #loc_loss = loc_loss / (batch_size * len(gt_delta))
        
        # debug - visualize
        if cfg.visualize & self.training:
            d_pred_scores = torch.cat(pred_scores[0], dim=0)
            d_pred_deltas = torch.cat(pred_deltas[0], dim=0)

            idxs = torch.where(anchor_labels[0] == 1)[0]
            pos_anchors = anchors[idxs]
            pos_deltas = d_pred_deltas[idxs]
            d_match_gt_boxes = match_gt_boxes[0][idxs]

            pos_proposal = Box.delta_to_pos(pos_anchors, pos_deltas)
            visualize_labeled_box(self.img, d_match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_anchor_image.jpg')

            scores, idx = torch.sort(d_pred_scores, dim=0, descending=True)
            scores, topk_idx = scores[:50], idx[:50]
            d_delta = d_pred_deltas[topk_idx]
            d_anchors = anchors[topk_idx]

            pos_proposal = Box.delta_to_pos(d_anchors, d_delta)
            visualize_labeled_box(self.img, d_match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_score_image.jpg')

        return cls_loss, loc_loss

    def get_proposals(self, total_anchors, batch_pred_scores, batch_pred_deltas, images):
        img_size = (images[0].shape[1], images[0].shape[2])
        '''
            total_anchors :       [N, 4]
            batch_pred_scores :   [bs, 5, n]
            batch_pred_deltas :   [bs, 5, n, 4]
        '''

        result_proposals =[]

        for pred_scores, pred_deltas in zip(batch_pred_scores, batch_pred_deltas):
            pred_proposals = []
            proposal_scores = []

            for scores, deltas, anchors in zip(pred_scores, pred_deltas, total_anchors):
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
            
            # reshuffling
            '''
            idx = torch.randperm(topk_idx.numel())
            topk_idx = topk_idx[idx]
            '''

            result_proposals.append(pred_proposals[topk_idx])
            
        if cfg.visualize:
            visualize_box(self.img, result_proposals[0][:80], './outputs/proposal_image.jpg')
        
        return result_proposals
