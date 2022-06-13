import torch
import torch.nn as nn
import os
import cv2
import matplotlib.image as mpimg
import numpy
import numpy as np
import time
from datasets.crowd import Crowd

# from models.vgg import vgg19
# from models.CSRNet_model import CSRNet
# from models.RFB_E_models import RFB_CNet
# from models.CCNet_model import CCNet
# from models.SK_model import SKNet
# from models.SKCCNet_model import SKCCNet
# from models.SKCount_model import SKCount
# from models.pyconv_model import PyConvNet
# from models.pyconvgg_model import PyConvggNet
# from models.eca_model import ECANet
# from models.PyConvECNet_model import PyConvECNet
from models.PyConv_ECA_vgg import PyConv_ECA_vgg19

import argparse

args = None

save_output = True
def save_density_map(density_map,output_dir, fname='results.png'):
    np.seterr(divide='ignore', invalid='ignore')
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    # mpimg.imsave(os.path.join(output_dir,fname),density_map)

output_dir = '/opt/data/nfs/gaoguangshuai/BL/DroneCrowd/vgg_density_maps/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    ############for UCF-QNRF############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF-QNRF/UCF-QNRF-Train-Val-Test/',
    #                         help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF-QNRF/save_models/1122-131949/',
    #                         help='model directory')
    # ############for SHT_A############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_A/SHT_A-Train-Val-Test/',
    #                         help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_A/save_models/0601_006666/',
    #                         help='model directory')
    ############for SHT_B############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_B/SHT_B-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/shanghaiTech_B/save_models/0830-222938/',
    #                     help='model directory')

    ############for UCF_CC_50############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF_CC_50/UCF_CC_50-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/UCF_CC_50/save_models/0901-193137/',
    #                     help='model directory')
    ############for building############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/building/building-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/building/save_models/0629-182843/',
    #                     help='model directory')
    ############# for small-vehicle############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/small-vehicle/small-vehicle-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/small-vehicle/save_models/0826-221826/',
    #                     help='model directory')
    ###########for large-vehicle############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/large-vehicle/large-vehicle-Train-Val-Test/',
    #                     help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/large-vehicle/save_models/0826-222247/',
    #                     help='model directory')
    ############# for ship############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/ship/ship-Train-Val-Test/',
    #                         help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/ship/save_models/0826-221907/',
    #                         help='model directory')

    #############for CARPK############
    # parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/CARPK/CARPK-Train-Val-Test/',
    #                         help='training data directory')
    # parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/CARPK/save_models/0702-102059/',
    #                         help='model directory')

    #############for DroneCrowd############
    parser.add_argument('--data-dir', default='/opt/data/nfs/gaoguangshuai/BL/DroneCrowd/DroneCrowd-Train-Val-Test/',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/opt/data/nfs/gaoguangshuai/BL/DroneCrowd/save_models/0702-205111/', #0702-213618
                        help='model directory')

    parser.add_argument('--device', default='3', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    print('args.data_dir:',args.data_dir)

    datasets = Crowd(os.path.join(args.data_dir, 'test_crowd'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    # model = vgg19()
    # model = CSRNet()
    # model = RFB_CNet()
    # model = CCNet()
    # model = SKNet()
    # model = SKCCNet()
    # model = SKCount()
    # model = PyConvNet()
    # model = PyConvggNet()
    # model = ECANet()
    # model = PyConvECNet()
    model = PyConv_ECA_vgg19()
    # model = nn.DataParallel(model)

    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    f = open('filename.txt', 'w')
    gt_f = open('gt_count.txt', 'w')
    pre_f = open('pre_count.txt', 'w')

    begin_time = time.time()

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        print(name)
        name = str(name)
        f.write(name)
        f.write('\n')
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.data.cpu().numpy()
            gt_count = count[0].item()
            print('gt_count:',gt_count)

            gt_f.writelines(str(gt_count))
            gt_f.write('\n')
            # pre_count = torch.sum(outputs).item()
            pre_count = np.sum(outputs)
            print('pre_count:', pre_count)
            pre_f.write(str(pre_count))
            pre_f.write('\n')

            temp_minu = gt_count - pre_count
            print(name, temp_minu, gt_count, pre_count)
            print('**************************************************************')

            epoch_minus.append(temp_minu)

            if save_output:
                save_density_map(outputs, output_dir, 'output_' + name.split('.')[0] + '.png')
    end_time = time.time()
    res_time = begin_time - end_time
    print('res_time:',res_time)

    f.close()
    gt_f.close()
    pre_f.close()

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
