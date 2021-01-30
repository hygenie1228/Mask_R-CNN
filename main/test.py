import argparse
import torch
import torch.backends.cudnn as cudnn
import time

from config import cfg
from base import Tester
from utils.logger import Logger


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

    # set tester
    tester = Tester()
    tester.build_dataloader()
    tester.load_model()

    # logger
    logger = Logger()

    for i, data in enumerate(trainer.dataloader):   

        trainer.optimizer.zero_grad()
        cls_loss, loc_loss = trainer.model(data)

        loss = cls_loss + loc_loss * 5.0
        loss.backward()
        trainer.optimizer.step()

        logger.log(cls_loss, loc_loss, epoch, i, epoch*len(trainer.dataloader)+i)


if __name__ == "__main__":
    main()