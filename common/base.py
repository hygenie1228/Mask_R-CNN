import torch
import cv2
from torch.utils.data import DataLoader

from dataset import DatasetManager
from nets.rcnn import MaskRCNN
from config import cfg
import copy

class Trainer:
    def __init__(self):
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.lr = None

    def build_dataloader(self):
        dataset = DatasetManager(cfg.train_datasets, train=cfg.is_train)
        
        self.dataloader = DataLoader(
            dataset,
            batch_size = cfg.batch_size,
            num_workers = cfg.num_worker,
            shuffle = cfg.shuffle,
            collate_fn = dataset.collate_fn
        )

    def build_model(self):
        self.model = MaskRCNN()
        self.model.cuda()
        self.model.train()
        print(self.model)       

    def set_lr(self, epoch):
        if epoch == 0:
            self.lr = cfg.lr
        else: 
            self.lr = cfg.lr * 0.1

    def set_optimizer(self):
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value],'lr': self.lr, 'weight_decay': cfg.weight_decay}]

        #self.optimizer = torch.optim.Adam(params)
        self.optimizer = torch.optim.SGD(params, momentum=cfg.momentum)
        
        # freeze backbone & rpn model
        if cfg.freeze_rpn:
            self.model.freeze_rpn()

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, cfg.save_model_path)


    def load_model(self):
        self.build_model()
        self.set_optimizer()
        #
        
        checkpoint = torch.load(cfg.load_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1

        return epoch