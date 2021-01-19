import argparse
import torch
import torch.backends.cudnn as cudnn
import cv2

from config import cfg
from base import Trainer
from utils.visualize import visualize_input_image, visualize_labeled_anchors
import time
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()
   
    return args

def main():
    # argument parse and create log
    args = parse_args()

    # restrict one gpu (not support distributed learning)
    cfg.set_args(args.gpu)
    cudnn.benchmark = True

    # set trainer
    trainer = Trainer()
    trainer.build_dataloader()
    if cfg.load_checkpoint:
        start_epoch = trainer.load_model()
    else :
        trainer.build_model()
        trainer.set_optimizer()
        start_epoch = 0
    
    start_time = time.time()
    writer = SummaryWriter('../runs')

    cls_losses = 0
    loc_losses = 0
    for epoch in range(start_epoch, cfg.epoch):
        for i, data in enumerate(trainer.dataloader):
            gt_data = data[0]['gt_data']
            raw_gt_data = data[0]['raw_gt_data']

            trainer.optimizer.zero_grad()

            cls_loss, loc_loss = trainer.model(data)

            loss = cls_loss + 10.0 * loc_loss
            loss.backward()
            trainer.optimizer.step()
            
            cls_losses = cls_losses + cls_loss
            loc_losses = loc_losses + loc_loss

            if i % 50 == 49:
                writer.add_scalar('Train_Loss/cls_loss', cls_losses.item(), i)
                writer.add_scalar('Train_Loss/loc_loss', loc_losses.item(), i)
                print("Epoch: %d / Iter : %d / cls Loss : %f / loc Loss : %f / Time : %f "%(epoch, i, cls_losses, loc_losses, time.time() - start_time))
                cls_losses = 0
                loc_losses = 0

            '''
            if cfg.visualize & (cfg.is_train != 'test') :  
                img = data[0]['image']
                img = img.numpy().transpose(1, 2, 0)
                visualize_input_image(img, gt_data, './outputs/input_image.jpg')

                gt_img = data[0]['raw_image']
                visualize_input_image(gt_img, raw_gt_data, './outputs/gt_image.jpg')
            '''

        if cfg.save_checkpoint & (epoch % 50 == 0):
            trainer.save_model(epoch)

if __name__ == "__main__":
    main()