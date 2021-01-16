import os
import sys

class Config:
    # path config
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    model_path = os.path.join(root_dir, 'checkpoint.pth')

    # datasets
    train_datasets = ['COCOKeypoint']
    test_datasets = ['COCOKeypoint']

    # preprocessing
    pixel_mean = [103.530, 116.280, 123.675]
    pixel_std = [1.0, 1.0, 1.0]
    min_size = 800
    max_size = 1333
    pad_unit = 32
    
    # FPN
    resnet_pretrained = True
    fpn_pretrained = False

    # RPN
    rpn_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    anchor_strides = [4, 8, 16, 32, 64]
    anchor_scales = [32, 64, 128, 256, 512]
    anchor_ratios = [0.5, 1.0, 2.0]
    pre_nms_topk_train = 12000
    pre_nms_topk_test = 6000
    post_nms_topk_train = 2000
    post_nms_topk_test = 2000
    negative_anchor_threshold = 0.3
    positive_anchor_threshold = 0.7
    positive_ratio = 0.5
    nms_threshold = 0.7

    # training parameter
    checkpoint = False
    batch_size = 1
    num_worker = 1
    shuffle = False
    epoch = 1000
    is_train = 'train'
    visualize = True
    lr = 0.0005
    weight_decay = 0.0001
    momentum = 0.9
    smooth_l1_beta = 0.5

    def set_args(self, gpu):
        self.gpu = gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        print('>>> Using GPU: {}'.format(self.gpu))

cfg = Config()
sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
sys.path.insert(0, os.path.join(cfg.root_dir, 'data'))