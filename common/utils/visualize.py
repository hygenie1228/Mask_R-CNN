import cv2
import numpy as np

from utils.func import Box
from utils.func import Keypoint

def visualize_input_image(img, gt_boxes, output_path):
    img = Box.visualize_box(img, gt_boxes, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_path, img)

def visualize_result(img, results, gt_boxes, output_path):
    boxes = []
    for result in results:
        boxes.append(result['bbox'])
    boxes = np.array(boxes)

    img = Box.visualize_box(img, gt_boxes, color=(0, 200, 0), thickness=2)
    img = Box.visualize_box(img, boxes, color=(255, 255, 255), thickness=2)
    cv2.imwrite(output_path, img)

def visualize_box(img, boxes, output_path):
    img = img.cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(output_path, img)
    img = cv2.imread(output_path)
        
    img = Box.visualize_box(img, boxes, color=(255, 255, 255), thickness=1)
    cv2.imwrite(output_path, img)

def visualize_labeled_box(img, bboxs, pos_boxes, neg_boxes, output_path):
    img = img.cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(output_path, img)
    img = cv2.imread(output_path)
    
    img = Box.visualize_box(img, bboxs, color=(0, 255, 0), thickness=3)
    img = Box.visualize_box(img, neg_boxes, color=(0, 0, 255), thickness=1)
    img = Box.visualize_box(img, pos_boxes, color=(255, 0, 0), thickness=1)
    
    cv2.imwrite(output_path, img)
