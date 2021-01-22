import numpy as np
import torch
import cv2

class Img:
    def resize_img(img, min_size, max_size):
        h, w, _ = img.shape
        new_h, new_w = h, w

        if new_h < new_w:
            if new_h < min_size:
                new_w = int(w * min_size / h)
                new_h = min_size
            if new_w > max_size:
                new_h = int(new_h * max_size / new_w)
                new_w = max_size
        else:
            if new_w < min_size:
                new_h = int(h * min_size / w)
                new_w = min_size
            if new_h > max_size:
                new_w = int(new_w * max_size / new_h)
                new_h = max_size
        
        scale_h = new_h / h
        scale_w = new_w / w
        return cv2.resize(img, (new_w, new_h)), (scale_h, scale_w)

    def normalize_img(img, pixel_mean, pixel_std):
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        pixel_mean = torch.Tensor(pixel_mean).reshape(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).reshape(-1, 1, 1)
        
        return (img - pixel_mean) / pixel_std

    def padding_img(img, pad_unit):
        _, h, w = img.shape
        
        new_h = (h + (pad_unit - 1)) // pad_unit * pad_unit
        new_w = (w + (pad_unit - 1)) // pad_unit * pad_unit
        
        padded_img = torch.zeros((3, new_h, new_w))
        padded_img[:, :h, :w] = img

        return padded_img

class Box:
    def xywh_to_xyxy(box):
        xmin = box[0]
        ymin = box[1]
        xmax = xmin + box[2]
        ymax = ymin + box[3]

        return [xmin, ymin, xmax, ymax]

    def scale_box(boxs, scales):
        new_boxs = []

        for box in boxs:
            new_boxs.append([box[0]*scales[1], box[1]*scales[0], box[2]*scales[1], box[3]*scales[0]])
        return new_boxs

    def calculate_iou_matrix(boxs1, boxs2):
        N, _ = boxs1.shape      # [N, 4]
        M, _ = boxs2.shape      # [M, 4]
        
        boxs1 = boxs1.view(N, 1, 4).expand(N, M, 4)     # [N, M, 4]
        boxs2 = boxs2.view(1, M, 4).expand(N, M, 4)     # [N, M, 4]

        iw_max = torch.min(boxs1[:, :, 2], boxs2[:, :, 2])
        iw_min = torch.max(boxs1[:, :, 0], boxs2[:, :, 0])
        iw = (iw_max - iw_min)
        iw[iw < 0] = 0

        ih_max = torch.min(boxs1[:, :, 3], boxs2[:, :, 3])
        ih_min = torch.max(boxs1[:, :, 1], boxs2[:, :, 1])
        ih = (ih_max - ih_min)
        ih[ih < 0] = 0
        
        boxs1_area = ((boxs1[:, :, 2] - boxs1[:, :, 0]) * (boxs1[:, :, 3] - boxs1[:, :, 1]))
        boxs2_area = ((boxs2[:, :, 2] - boxs2[:, :, 0]) * (boxs2[:, :, 3] - boxs2[:, :, 1]))

        inter = iw * ih                             # [N, M]
        union = boxs1_area + boxs2_area - inter     # [N, M]
        iou_matrix = inter / union                  # [N, M]

        return iou_matrix

    def visualize_box(img, boxs, color, thickness):
        for box in boxs:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            
            pos1 = (x_min, y_min)
            pos2 = (x_min, y_max)
            pos3 = (x_max, y_min)
            pos4 = (x_max, y_max)
            
            img = cv2.line(img, pos1, pos2, color, thickness) 
            img = cv2.line(img, pos1, pos3, color, thickness) 
            img = cv2.line(img, pos2, pos4, color, thickness) 
            img = cv2.line(img, pos3, pos4, color, thickness) 

        return img

    def visualize_dot(img, boxs, color, thickness):
        for box in boxs:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            
            pos = (int((x_min + x_max)/2), int((y_min + y_max)/2)) 
            img = cv2.line(img, pos, pos, color, 7) 

        return img
    
    def pos_to_delta(box1, box2):
        # gt_box
        box1_w = box1[:, 2] - box1[:, 0]
        box1_h = box1[:, 3] - box1[:, 1]
        box1_ctr_x = box1[:, 0] + 0.5 * box1_w
        box1_ctr_y = box1[:, 1] + 0.5 * box1_h

        # anchor
        box2_w = box2[:, 2] - box2[:, 0]
        box2_h = box2[:, 3] - box2[:, 1]
        box2_ctr_x = box2[:, 0] + 0.5 * box2_w
        box2_ctr_y = box2[:, 1] + 0.5 * box2_h        

        dx = (box1_ctr_x - box2_ctr_x) / box2_w
        dy = (box1_ctr_y - box2_ctr_y) / box2_h
        dw = torch.log(box1_w / box2_w)
        dh = torch.log(box1_h / box2_h)

        return torch.stack((dx, dy, dw, dh), dim=1)


    def delta_to_pos(boxes, delta):
        # anchor
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        box_x = boxes[:, 0] + 0.5 * box_w
        box_y = boxes[:, 1] + 0.5 * box_h

        # pred delta
        dx = delta[:, 0]
        dy = delta[:, 1]
        dw = delta[:, 2]
        dh = delta[:, 3]

        pred_x = dx * box_w + box_x
        pred_y = dy * box_h + box_y
        pred_w = torch.exp(dw) * box_w
        pred_h = torch.exp(dh) * box_h

        pos_x1 = pred_x - 0.5 * pred_w 
        pos_y1 = pred_y - 0.5 * pred_h
        pos_x2 = pred_x + 0.5 * pred_w
        pos_y2 = pred_y + 0.5 * pred_h

        return torch.stack((pos_x1, pos_y1, pos_x2, pos_y2), dim=1)

    def box_valid_check(scores, boxes, img_size):
        # finite check
        mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
        boxes = boxes[mask]
        scores = scores[mask]

        H, W = img_size

        # clamp 
        boxes[:, 0].clamp_(min=0, max=W)
        boxes[:, 1].clamp_(min=0, max=H)
        boxes[:, 2].clamp_(min=0, max=W)
        boxes[:, 3].clamp_(min=0, max=H)

        # removal
        '''
        idxs = torch.where((boxes[:, 0] >= 0) & (boxes[:, 0] <= W))[0]
        scores, boxes = scores[idxs], boxes[idxs]
        idxs = torch.where((boxes[:, 1] >= 0) & (boxes[:, 1] <= H))[0]
        scores, boxes = scores[idxs], boxes[idxs]
        idxs = torch.where((boxes[:, 2] >= 0) & (boxes[:, 2] <= W))[0]
        scores, boxes = scores[idxs], boxes[idxs]
        idxs = torch.where((boxes[:, 3] >= 0) & (boxes[:, 3] <= H))[0]
        scores, boxes = scores[idxs], boxes[idxs]
        '''
        
        # area check
        mask = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
        scores, boxes = scores[mask], boxes[mask]

        return scores, boxes

class Keypoint:
    def to_array(keypoints):
        if len(keypoints) == 0:
            return []
        
        num_objs = len(keypoints)
        num_keys = len(keypoints[0]) // 3
        return np.array(keypoints).reshape(num_objs, num_keys, 3)
    
    def scale_keypoint(keypoints, scales):
        new_keypoints = []

        for keypoint in keypoints:
            new_keypoint = [] 
            for key in keypoint:
                new_key = [key[0]*scales[1], key[1]*scales[0], key[2]]
                new_keypoint.append(new_key)
            new_keypoints.append(new_keypoint)
        
        return np.array(new_keypoints)

    def visualize_keypoint(img, keypoints, color):
        thickness = 5

        for keypoint in keypoints:
            for key in keypoint:
                if key[2] != 0:
                    pos = (int(key[0]), int(key[1]))
                    img = cv2.line(img, pos, pos, color, thickness) 

        return img
