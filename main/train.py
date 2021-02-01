import argparse
import torch
import torch.backends.cudnn as cudnn

from config import cfg
from base import Trainer
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()
    return args

def main():
    # argument parse and create log
    args = parse_args()

    # restrict one gpu : not support distributed learning
    cfg.set_args(args.gpu)
    cudnn.benchmark = True

    # set trainer
    trainer = Trainer()
    trainer.build_dataloader()
    trainer.build_model()
    trainer.set_optimizer()
    start_epoch = 0

    # load model
    if cfg.load_checkpoint:
        start_epoch = trainer.load_model()

    # logger
    logger = Logger()

    # train model
    for epoch in range(start_epoch, cfg.epoch):
        for i, data in enumerate(trainer.dataloader):   
            trainer.optimizer.zero_grad()
            proposal_loss, detection_loss = trainer.model(data)

            loss = proposal_loss[0] + proposal_loss[1] + detection_loss[0] + detection_loss[1]
            loss.backward()
            trainer.optimizer.step()

            logger.log(proposal_loss, detection_loss, epoch, i, epoch*len(trainer.dataloader)+i)
        if cfg.save_checkpoint:
            trainer.save_model(epoch)

if __name__ == "__main__":
    main()