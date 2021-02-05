from torch.utils.tensorboard import SummaryWriter
import time

from config import cfg

class Logger:
    def __init__(self):
        self.writer = SummaryWriter('../runs')
        self.start_time = time.time()
        self.proposal_cls = 0
        self.proposal_loc = 0
        self.detection_cls = 0
        self.detection_loc = 0
    
    def log(self, proposal_loss, detection_loss, epoch, i, iters):
        self.proposal_cls = self.proposal_cls + proposal_loss[0].item()
        self.proposal_loc = self.proposal_loc + proposal_loss[1].item() * cfg.pro_loc_lambda
        self.detection_cls = self.detection_cls + detection_loss[0].item()
        self.detection_loc = self.detection_loc + detection_loss[1].item() * cfg.det_loc_lambda

        if i % cfg.log_interval == (cfg.log_interval-1):
            self.writer.add_scalar('Train_Loss/proposal_cls', self.proposal_cls, iters)
            self.writer.add_scalar('Train_Loss/proposal_loc', self.proposal_loc, iters)
            self.writer.add_scalar('Train_Loss/detection_cls', self.detection_cls, iters)
            self.writer.add_scalar('Train_Loss/detection_loc', self.detection_loc, iters)
            

            print("Epoch: %d /  Iter : %d /  Pro cls : %f \t/  Pro loc : %f \t/  Det cls : %f \t/  Det loc : %f \t/  Time : %f " \
                    %(epoch, i, self.proposal_cls, self.proposal_loc, self.detection_cls, self.detection_loc, time.time() - self.start_time))

            self.proposal_cls = 0
            self.proposal_loc = 0
            self.detection_cls = 0
            self.detection_loc = 0