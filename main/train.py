import argparse
import torch
import torch.backends.cudnn as cudnn
import cv2

from config import cfg
from base import Trainer
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
        start_epoch = 0
    
    writer = SummaryWriter('../runs')

    start_time = time.time()

    trainer.set_lr(0)
    trainer.set_optimizer()

    for epoch in range(start_epoch, cfg.epoch):
        cls_losses = 0
        loc_losses = 0
        for i, data in enumerate(trainer.dataloader):   
            trainer.optimizer.zero_grad()

            cls_loss, loc_loss = trainer.model(data)

            loss = cls_loss + loc_loss
            loss.backward()
            trainer.optimizer.step()
            
            cls_losses = cls_losses + cls_loss
            loc_losses = loc_losses + loc_loss
            if i % 100 == 99:
                writer.add_scalar('Train_Loss/cls_loss', cls_losses.item(), epoch * len(trainer.dataloader) + i)
                writer.add_scalar('Train_Loss/loc_loss', loc_losses.item(), epoch * len(trainer.dataloader) + i)
                print("Epoch: %d / Iter : %d / cls Loss : %f / loc Loss : %f / Time : %f "%(epoch, i, cls_losses, loc_losses, time.time() - start_time))
                cls_losses = 0
                loc_losses = 0

        if cfg.save_checkpoint:
            trainer.save_model(epoch)

if __name__ == "__main__":
    main()