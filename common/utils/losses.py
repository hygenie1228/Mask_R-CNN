import torch

def smooth_l1_loss(bbox_pred, bbox_targets, beta=1.0):
    box_diff = bbox_pred - bbox_targets
    abs_in_box_diff = torch.abs(box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()

    loss_box = smoothL1_sign * 0.5 * torch.pow(box_diff, 2) / beta + \
                (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))

    loss_box = loss_box.view(-1).sum(0)
    return loss_box