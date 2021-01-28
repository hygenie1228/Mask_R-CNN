import torch
from torch import nn
import torch.nn.functional as F

from config import cfg
from nets.roi_align import ROIAlign
from utils.func import Box
from utils.losses import smooth_l1_loss
from utils.visualize import visualize_anchors, visualize_labeled_anchors

class ROIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = cfg.roi_threshold
        self.num_sample = cfg.roi_num_sample
        self.positive_ratio = cfg.roi_positive_ratio
        self.num_labels = cfg.num_labels
        self.nms_threshold = cfg.roi_head_nms_threshold
        
        # roi align layer
        self.roi_align = ROIAlign()

        # layers
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(in_features=256*7*7, out_features=1024, bias=True)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.fc_relu2 = nn.ReLU()

        # layer - classifier
        self.cls_score = nn.Linear(in_features=1024, out_features=self.num_labels+1, bias=True)
        self.bbox_pred = nn.Linear(in_features=1024, out_features=4, bias=True)

        self.img = None

    def forward(self, features, proposals, images, gt_datas):
        # for debugging
        self.img = images[0]

        if self.training:
            gt_labels = [torch.tensor(gt_data['category_id']).cuda() for gt_data in gt_datas]
            gt_boxes = [torch.tensor(gt_data['bboxs']).cuda() for gt_data in gt_datas]

            # add gt_box to proposals
            reshape_proposals = self.add_gt_boxes(proposals, gt_boxes)

            # labeling
            proposal_labels, match_gt_boxes = self.labeling_proposals(reshape_proposals, gt_labels, gt_boxes)
            proposal_labels = self.sampling_proposals(proposal_labels)
            proposal_labels, reshape_proposals, match_gt_boxes, num_features = self.reassign_proposals(proposal_labels, reshape_proposals, match_gt_boxes)

            # roi align layer
            align_features = self.roi_align(features, reshape_proposals)
            
            # predict scores, deltas
            pred_scores, pred_deltas = self.predict_scores_deltas(align_features)
            
            cls_loss, loc_loss = self.loss(reshape_proposals, pred_scores, pred_deltas, proposal_labels, match_gt_boxes, num_features)
        
        # Inference part
        # roi align layer
        align_features = self.roi_align(features, proposals)
        # predict scores, deltas
        pred_scores, pred_deltas = self.predict_scores_deltas(align_features)

        # get top detections
        results = get_top_detections(proposals, pred_scores, pred_deltas)

        if cfg.visualize & self.training:   
            index = 0
            idxs = torch.where(proposal_labels[index] >= 1)[0]
            pos_proposals = reshape_proposals[index][idxs]
            gt_boxes = match_gt_boxes[index][idxs]
            idxs = torch.where(proposal_labels[index] == 0)[0]
            neg_proposals = reshape_proposals[index][idxs] 
            visualize_labeled_anchors(self.img, gt_datas[index]['bboxs'], pos_proposals, pos_proposals, './outputs/labeled_proposal_image.jpg')

        return cls_loss, loc_loss


    def get_top_detections(self, proposals, pred_scores, pred_deltas):
        start_idx = 0

        for proposal in proposals:
            pred_score = pred_scores[start_idx : start_idx + len(proposal)]
            pred_delta = pred_deltas[start_idx : start_idx + len(proposal)]
            start_idx = start_idx + len(proposal)



            # pre nms 
            # decoding
            # valid box
            # nms


    def predict_scores_deltas(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)

        pred_scores = self.cls_score(x)
        print(pred_scroes.shape)
        pred_scores = F.softmax(pred_scores, dim=1)
        pred_deltas = self.bbox_pred(x)

        return pred_scores, pred_deltas

    def loss(self, proposals, pred_scores, pred_deltas, proposal_labels, match_gt_boxes, num_features):
        start_idx = 0
        cls_loss = 0.0
        loc_loss = 0.0
        batch_size = len(proposal_labels)

        for proposal, gt_label, gt_box, interval in zip(proposals, proposal_labels, match_gt_boxes, num_features):
            pred_score = pred_scores[start_idx : start_idx + interval]
            pred_delta = pred_deltas[start_idx : start_idx + interval]
            start_idx = start_idx + interval

            # score loss function
            cls_loss = cls_loss + F.cross_entropy(pred_score, gt_label.long(), reduction="sum")
            
            # deltas matching
            idxs = torch.where(gt_label > 0)[0]
            pos_proposal = proposal[idxs]                                   # proposal
            pos_gt_boxes = gt_box[idxs]                                     # gt boxes

            gt_deltas = Box.pos_to_delta(pos_gt_boxes, pos_proposal)        # target
            pos_deltas = pred_delta[idxs]                                   # pred

            # delta loss function
            loc_loss = loc_loss + smooth_l1_loss(pos_deltas, gt_deltas, beta=cfg.smooth_l1_beta)

        # normalizer
        cls_loss = cls_loss / (batch_size * self.num_sample)
        loc_loss = loc_loss / (batch_size * self.num_sample)

        if cfg.visualize & self.training:
            # for visualize - trianing
            d_pred_scores = pred_scores[:num_features[0]]
            d_pred_deltas = pred_deltas[:num_features[0]]

            idxs = torch.where(proposal_labels[0] == 1)[0]
            pos_proposals = proposals[0][idxs]
            pos_deltas = d_pred_deltas[idxs]
            d_match_gt_boxes = match_gt_boxes[0][idxs]

            pos_proposal = Box.delta_to_pos(pos_proposals, pos_deltas)
            visualize_labeled_anchors(self.img, d_match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_proposal_image.jpg')

            scores, idx = d_pred_scores[:, 1].sort(descending=True)
            scores, topk_idx = scores[:15], idx[:15]
            d_delta = d_pred_deltas[topk_idx]
            d_proposals = proposals[0][topk_idx]

            pos_proposal = Box.delta_to_pos(d_proposals, d_delta)
            visualize_labeled_anchors(self.img, d_match_gt_boxes, pos_proposal, pos_proposal, './outputs/debug_final_image.jpg')
            

        return cls_loss, loc_loss

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

            sampling_pos_num = min(pos_index.numel(), self.num_sample * self.positive_ratio)
            sampling_neg_num = min(neg_index.numel(), self.num_sample * (1 - self.positive_ratio))

            rand_idx = torch.randperm(pos_index.numel())[:int(sampling_pos_num)]
            pos_index = pos_index[rand_idx]
            rand_idx = torch.randperm(neg_index.numel())[:int(sampling_neg_num)]
            neg_index = neg_index[rand_idx]
            
            sampling_label[pos_index] = labels[pos_index]
            sampling_label[neg_index] = 0

            sampling_labels.append(sampling_label)

        return sampling_labels

    def add_gt_boxes(self, proposals, gt_boxes):
        new_proposals = []

        for proposal, gt_box in zip(proposals, gt_boxes):
            new_proposals.append(torch.cat([proposal, gt_box], dim=0))
        
        proposals = []
        for boxes in new_proposals:
            mask = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
            boxes = boxes[mask]
            proposals.append(boxes)

        return proposals

    def reassign_proposals(self, proposal_labels, proposals, match_gt_boxes):
        new_labels = []
        new_proposals = []
        new_match_gt_boxes = []
        num_features = []

        for label, proposal, match_boxes in zip(proposal_labels, proposals, match_gt_boxes):
            idxs = torch.where(label >= 0)[0]
            new_labels.append(label[idxs])
            new_proposals.append(proposal[idxs])
            new_match_gt_boxes.append(match_boxes[idxs])
            num_features.append(len(idxs))

        return new_labels, new_proposals, new_match_gt_boxes, num_features