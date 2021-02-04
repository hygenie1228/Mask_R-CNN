import os
import cv2
import torch
import numpy as np
import copy

from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask

from config import cfg
from utils.func import Img
from utils.func import Box
from utils.func import Keypoint

class COCOKeypointDataset(Dataset):
    def __init__(self, train='train'):
        # set annotation & data path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cur_dir = os.path.join(cur_dir, '.')
        if train == 'train':
            self.annot_path = os.path.join(cur_dir, 'dataset', 'annotations', 'person_keypoints_train2017.json')
            self.root = os.path.join(cur_dir, 'dataset', 'train2017')
        elif train == 'val':
            self.annot_path = os.path.join(cur_dir, 'dataset', 'annotations', 'person_keypoints_val2017.json')
            self.root = os.path.join(cur_dir, 'dataset', 'val2017')
        elif train == 'test':
            self.annot_path = os.path.join(cur_dir, 'dataset', 'annotations', 'image_info_test2017.json')
            self.root = os.path.join(cur_dir, 'dataset', 'test2017')
        
        # dataset type
        if train in ['train', 'val']:
            self.is_train = True
        else:
            self.is_train = False

        # load data
        self.db = COCO(self.annot_path)

        self.img_ids = self.db.getImgIds(catIds = 1)        # only get person image id
        self.img_ids = list(sorted(self.img_ids))
        self.annots = []
        self.img_paths = []
        self.gt_datas = []
        for img_id in self.img_ids:
            # get annotations
            ann_ids = self.db.getAnnIds(imgIds=img_id)
            anns = self.db.loadAnns(ann_ids)
            
            # get image path
            path = self.db.loadImgs(img_id)[0]['file_name']

            category_ids = []
            bboxs = []
            keypoints = []
            num_keypoints = []
            areas = []
            iscrowds = []
            for ann in anns:
                category_ids.append(ann['category_id'])
                bboxs.append(Box.xywh_to_xyxy(ann['bbox']))
                keypoints.append(ann['keypoints'])
                num_keypoints.append(ann['num_keypoints'])
                areas.append(ann['area'])
                iscrowds.append(ann['iscrowd'])
                #bimask = self.db.annToMask(ann)
                #segmentations.append(0)
            
            self.annots.append(anns)
            self.img_paths.append(path)
            if self.is_train:
                self.gt_datas.append({
                    'category_id' : category_ids,
                    'bboxs' : bboxs,
                    'keypoints' : Keypoint.to_array(keypoints),
                    'num_keypoints' : num_keypoints,
                    'areas' : areas,
                    'iscrowds' : iscrowds,
                    'segmentations' : None
                })
            else:
                self.gt_datas.append({})
        
        # init cv2 threads
        cv2.setNumThreads(0)
        print(len(self.img_ids))
    
    def __len__(self):
        #return 10
        return len(self.img_ids)

    def __getitem__(self, index): 
        #index = 1
        img_id = self.img_ids[index]
        anns = self.annots[index]
        img_path = self.img_paths[index]
        raw_gt_data = self.gt_datas[index]
        gt_data = copy.deepcopy(raw_gt_data)

        raw_img = cv2.imread(os.path.join(self.root, img_path))
        img, gt_data, img_size = self.preprocessing(raw_img, gt_data)

        return {
            'img_id' : img_id,
            'raw_image' : raw_img,
            'image' : img,
            'raw_gt_data' : raw_gt_data,
            'gt_data' : gt_data,
            'raw_img_size' : (raw_img.shape[0], raw_img.shape[1]),
            'img_size' : (img_size[1], img_size[2])
        }

    def preprocessing(self, img, gt_data):
        # resize image
        img, scales = Img.resize_img(img, cfg.min_size, cfg.max_size)
        # normalize image
        img = Img.normalize_img(img, cfg.pixel_mean, cfg.pixel_std)
        img_size = img.shape
        # padding image
        img = Img.padding_img(img, cfg.pad_unit)

        # recompose ground truth
        if self.is_train:
            gt_data['bboxs'] = Box.scale_box(gt_data['bboxs'], scales)
            gt_data['keypoints'] = Keypoint.scale_keypoint(gt_data['keypoints'], scales)
        return img, gt_data, img_size

    def evaluate(self):
        results = self.db.loadRes(cfg.save_result_path)

        coco_eval = COCOeval(self.db, results, 'bbox')
        coco_eval.params.catIds = [1]
        coco_eval.params.imgIds = self.img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()