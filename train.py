import torch
torch.set_num_threads(4)
from utils.regression_trainer import RegTrainer
import argparse
import os
import numpy as np
from random import seed
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF-QNRF/UCF-QNRF-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF-QNRF/save_models/',
    #                     help='directory to save models.')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_A/SHT_A-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_A/save_models/',
    #                     help='directory to save models.')
    #
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_B/SHT_B-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_B/save_models/',
    #                     help='directory to save models.')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF_CC_50/UCF_CC_50-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF_CC_50/save_models/',
    #                     help='directory to save models.')


    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/building/building-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/building/save_models/',
    #                     help='directory to save models.')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/small-vehicle/small-vehicle-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/small-vehicle/save_models/',
    #                     help='directory to save models.')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/large-vehicle/large-vehicle-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/large-vehicle/save_models/',
    #                     help='directory to save models.')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/ship/ship-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/ship/save_models/',
    #                     help='directory to save models.')

    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/CARPK/CARPK-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/CARPK/save_models/',
    #                     help='directory to save models.')

    parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/DroneCrowd/DroneCrowd-Train-Val-Test/',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/DroneCrowd/save_models/',
                        help='directory to save models.')

    parser.add_argument('--lr', type=float, default=1e-5,                         #1e-5
                        help='the initial learning rate')

    # parser.add_argument('--lr', type=float, default=1e-6,  # 1e-5
    #                     help='the initial learning rate')

    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000,                        #1000
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')

    parser.add_argument('--val-start', type=int, default=200,                     #200
                        help='the epoch start to val')

    parser.add_argument('--save-all-best', type=bool, default=True,
                        help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='train batch size')

    parser.add_argument('--device', default='0,3,4,5', help='assign device')

    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')

    ##### for UCF-QNRF,shanghai_B,small-vehicle,large-vehicle and ship dataset
    parser.add_argument('--crop-size', type=int, default=512,
                      help='the crop size of the train image')

    ##### for shanghai_A and building dataset
    # parser.add_argument('--crop-size', type=int, default=256,
    #                     help='the crop size of the train image')

    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')

    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')

    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')

    # parser.add_argument('--background-ratio', type=float, default=0.1,
    #                     help='background ratio')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()