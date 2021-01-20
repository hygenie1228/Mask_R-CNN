import os
import cv2
import torch
import numpy as np
import copy

from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from pycocotools import mask

from config import cfg
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
        return 10000
        #return len(self.img_ids)

    def __getitem__(self, index): 
        #index = 1
        img_id = self.img_ids[index]
        anns = self.annots[index]
        img_path = self.img_paths[index]
        raw_gt_data = self.gt_datas[index]
        gt_data = copy.deepcopy(raw_gt_data)

        raw_img = cv2.imread(os.path.join(self.root, img_path))
        img, gt_data = self.preprocessing(raw_img, gt_data)

        return {
            'img_id' : img_id,
            'raw_image' : raw_img,
            'image' : img,
            'raw_gt_data' : raw_gt_data,
            'gt_data' : gt_data,
            'raw_img_size' : (raw_img.shape[0], raw_img.shape[1]),
            'img_size' : (img.shape[1], img.shape[2])
        }

    def preprocessing(self, img, gt_data):
        # resize image
        img, scales = self.resize_img(img)
        # normalize image
        img = self.normalize_img(img)
        # padding image
        img = self.padding_img(img)

        # recompose ground truth
        if self.is_train:
            gt_data['bboxs'] = Box.scale_box(gt_data['bboxs'], scales)
            gt_data['keypoints'] = Keypoint.scale_keypoint(gt_data['keypoints'], scales)
        return img, gt_data
    
    def resize_img(self, img):
        h, w, _ = img.shape
        new_h, new_w = h, w

        if new_h < new_w:
            if new_h < cfg.min_size:
                new_w = int(w * cfg.min_size / h)
                new_h = cfg.min_size
            if new_w > cfg.max_size:
                new_h = int(new_h * cfg.max_size / new_w)
                new_w = cfg.max_size
        else:
            if new_w < cfg.min_size:
                new_h = int(h * cfg.min_size / w)
                new_w = cfg.min_size
            if new_h > cfg.max_size:
                new_w = int(new_w * cfg.max_size / new_h)
                new_h = cfg.max_size
        
        scale_h = new_h / h
        scale_w = new_w / w
        return cv2.resize(img, (new_w, new_h)), (scale_h, scale_w)

    def normalize_img(self, img):
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        pixel_mean = torch.Tensor(cfg.pixel_mean).reshape(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.pixel_std).reshape(-1, 1, 1)
        
        return (img - pixel_mean) / pixel_std

    def padding_img(self, img):
        _, h, w = img.shape
        
        new_h = (h + (cfg.pad_unit - 1)) // cfg.pad_unit * cfg.pad_unit
        new_w = (w + (cfg.pad_unit - 1)) // cfg.pad_unit * cfg.pad_unit
        
        padded_img = torch.zeros((3, new_h, new_w))
        padded_img[:, :h, :w] = img

        return padded_img
