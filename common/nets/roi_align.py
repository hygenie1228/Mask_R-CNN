import torch
from torch import nn

from config import cfg

class ROIAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = cfg.output_size
        self.pooler_scales = cfg.pooler_scales
        self.roi_features = cfg.roi_features

        # max pooling layer
        self.max_pooling = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, features, proposals):
        spatial_levels = self.calculate_level(proposals)

        bs = features[self.roi_features[0]].shape[0]
        reshape_features = []

        for i in range(bs):
            one_batch_feature = []
            for j in self.roi_features:
                one_batch_feature.append(features[j][i])
            reshape_features.append(one_batch_feature)

        for features_i, proposals_i, spatial_levels_i in zip(reshape_features, proposals, spatial_levels):
             self.extract_features(features_i, proposals_i, spatial_levels_i)
             break
    
    def extract_features(self, features, proposals, spatial_levels):
        output_features = []
        print(proposals.shape)
        print(spatial_levels.shape)
        for proposal, spatial_level in zip(proposals, spatial_levels):
            # get feature level
            feature = features[spatial_level]

            # rescaling
            proposal = proposal * self.pooler_scales[spatial_level]

            print("!!!")
            print(feature.shape)

            # get sampling indexs : (H, W, 2)
            w_range, h_range = self.calculate_sampling_idxs(proposal)

            interpolate_feature = self.bilinear_interpolate(feature, w_range, h_range)
            interpolate_feature = interpolate_feature.unsqueeze(0)

            print(interpolate_feature.shape)
            
            output_feature = self.max_pooling(interpolate_feature)
            output_features.append(output_feature)

            print(output_feature.shape)
        
        print("!!!!!")
        print(len(output_features))
        #output_features = torch.cat(output_features, dim=0)
        #print(output_features.shape)

    def bilinear_interpolate(self, feature, x, y):
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        
        x0.clamp_(min=0, max=feature.shape[2]-1)
        x1.clamp_(min=0, max=feature.shape[2]-1)
        y0.clamp_(min=0, max=feature.shape[1]-1)
        y1.clamp_(min=0, max=feature.shape[1]-1)

        la = feature[:, y0, x0]
        lb = feature[:, y1, x0]
        lc = feature[:, y0, x1]
        ld = feature[:, y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        interpolate_feature = wa * la + wb * lb + wc * lc + wd * ld
        return interpolate_feature

    def calculate_sampling_idxs(self, proposal):
        w_stride = (proposal[2] - proposal[0]) / self.output_size[1]
        h_stride = (proposal[3] - proposal[1]) / self.output_size[0]

        w_unit = w_stride / torch.ceil(w_stride)
        h_unit = h_stride / torch.ceil(h_stride)

        w_sampling_num = int(self.output_size[1] * torch.ceil(w_stride))
        h_sampling_num = int(self.output_size[0] * torch.ceil(h_stride))

        w_range = torch.arange(0, w_sampling_num).cuda() * w_unit + w_unit/2 + proposal[0]
        h_range = torch.arange(0, h_sampling_num).cuda() * h_unit + h_unit/2 + proposal[1]

        w_range = w_range.reshape(1, -1).repeat(h_sampling_num, 1)
        h_range = h_range.reshape(-1, 1).repeat(1, w_sampling_num)

        return w_range, h_range

    def calculate_level(self, proposals):
        spatial_levels = []
        
        for proposal in proposals:
            proposal_w = proposal[:, 2] - proposal[:, 0]
            proposal_h = proposal[:, 3] - proposal[:, 1]
            areas = proposal_w * proposal_h
            roi_level = torch.floor(2 + torch.log2(torch.sqrt(areas)/224.0))
            roi_level = roi_level.clamp(0, 3).long()
            spatial_levels.append(roi_level)

        return spatial_levels

