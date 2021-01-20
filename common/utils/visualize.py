import cv2

from utils.func import Box
from utils.func import Keypoint

def visualize_input_image(img, gt_data, output_path):
    cv2.imwrite(output_path, img)
    img = cv2.imread(output_path)

    img = Box.visualize_box(img, gt_data['bboxs'], color=(0, 255, 0), thickness=2)
    img = Keypoint.visualize_keypoint(img, gt_data['keypoints'], color=(0, 0, 255))
    cv2.imwrite(output_path, img)

def visualize_anchors(img, anchors, output_path):
    img = img.cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(output_path, img)
    img = cv2.imread(output_path)
        
    img = Box.visualize_box(img, anchors, color=(255, 255, 255), thickness=1)
    cv2.imwrite(output_path, img)

def visualize_labeled_anchors(img, bboxs, pos_anchors, neg_anchors, output_path):
    img = img.cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(output_path, img)
    img = cv2.imread(output_path)
    
    img = Box.visualize_box(img, bboxs, color=(0, 255, 0), thickness=4)
    img = Box.visualize_box(img, neg_anchors, color=(0, 0, 255), thickness=2)
    img = Box.visualize_box(img, pos_anchors, color=(255, 0, 0), thickness=2)
    
    cv2.imwrite(output_path, img)
