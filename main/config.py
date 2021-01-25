import os
import sys

class Config:
    # Path Config
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    
    # Training Parameter
    save_checkpoint = False
    save_model_path = os.path.join(root_dir, 'common', 'nets', 'checkpoints', 'checkpoint_epoch2_0123.pth')
    load_checkpoint = True
    load_model_path = os.path.join(root_dir, 'common', 'nets', 'checkpoints', 'checkpoint_epoch2_0123.pth')
    visualize = True

    is_train = 'train'
    batch_size = 2
    num_worker = 1
    shuffle = False
    epoch = 3
    lr = 0.002
    weight_decay = 0.0001
    momentum = 0.9
    smooth_l1_beta = 1.0 

    # Datasets
    train_datasets = ['COCOKeypoint']
    test_datasets = ['COCOKeypoint']

    # Preprocessing
    pixel_mean = [103.530, 116.280, 123.675]
    pixel_std = [1.0, 1.0, 1.0]
    min_size = 800
    max_size = 1333
    pad_unit = 32

    # FPN
    resnet_pretrained = True
    fpn_pretrained = False
    
    # Anchor Generator
    anchor_strides = [4, 8, 16, 32, 64]
    anchor_scales = [32, 64, 128, 256, 512]
    anchor_ratios = [0.5, 1.0, 2.0]
    negative_anchor_threshold = 0.3
    positive_anchor_threshold = 0.7
    anchor_num_sample = 128
    anchor_positive_ratio = 0.5

    # RPN
    rpn_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    pre_nms_topk_train = 12000
    pre_nms_topk_test = 6000
    post_nms_topk_train = 1000
    post_nms_topk_test = 1000
    nms_threshold = 0.7

    # ROIHead
    roi_threshold = 0.5
    roi_num_sample = 256
    roi_positive_ratio = 0.25
    num_labels = 1

    # ROIAlign
    output_size = (7, 7)
    roi_features = ['p2', 'p3', 'p4', 'p5']
    pooler_scales = [0.25, 0.125, 0.0625, 0.03125]


    def set_args(self, gpu):
        self.gpu = gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        print('>>> Using GPU: {}'.format(self.gpu))


cfg = Config()
sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
sys.path.insert(0, os.path.join(cfg.root_dir, 'data'))