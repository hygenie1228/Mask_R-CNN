import argparse
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config import cfg
from base import Tester
from utils.visualize import visualize_input_image, visualize_result

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
    
    

    for data in tqdm(tester.dataloader):   
        results = tester.model(data)
        tester.save_results(results)

        # visualize input image
        if cfg.visualize:
            visualize_input_image(data[0]['raw_image'], data[0]['raw_gt_data']['bboxs'], './outputs/input_image.jpg')
            visualize_result(data[0]['raw_image'], results, data[0]['raw_gt_data']['bboxs'], './outputs/result_image.jpg')

    tester.save_jsons()
    tester.evaluate()

if __name__ == "__main__":
    main()