# CONFIG
import argparse

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='OV_Inference')
arg.add_argument('-m', '--model_path', type=str, default='inference', help="OpenVINO Model IR Path")
arg.add_argument("-d", "--device", type=str, default="CPU", choices=["CPU", "GPU"], help="Inference Device: CPU/GPU")

arg.add_argument('-c', '--configuration', type=str, default='val_lrru_mini_kitti.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)

# ENVIRONMENT SETTINGS
import os
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools

# BASIC PACKAGES
import time
import random
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import openvino as ov


# MODULES
from dataloaders.kitti_loader import KittiDepth
# from utility import *
from PIL import Image
import matplotlib.pyplot as plt
cm = plt.get_cmap('plasma')

# VARIANCES
sample_, output_ = None, None
metric_txt_dir = None

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)


def save_result(epoch, idx, sample, output, args):
    img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rgb = sample['rgb']
    rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb))
    rgb = sample['rgb'].data.numpy()
    dep = sample['dep'].data.numpy()
    gt = sample['dep'].data.numpy()
    pred = output[0]
    num_summary = gt.shape[0]
    if num_summary > args.num_summary:
        num_summary = args.num_summary

        rgb = rgb[0:num_summary, :, :, :]
        dep = dep[0:num_summary, :, :, :]
        gt = gt[0:num_summary, :, :, :]
        pred = pred[0:num_summary, :, :, :]

    rgb = np.clip(rgb, a_min=0, a_max=1.0)
    dep = np.clip(dep, a_min=0, a_max=args.max_depth)
    gt = np.clip(gt, a_min=0, a_max=args.max_depth)
    pred = np.clip(pred, a_min=0, a_max=args.max_depth)

    list_imgv, list_imgh = [], []
    for b in range(0, num_summary):
        rgb_tmp = rgb[b, :, :, :]
        dep_tmp = dep[b, 0, :, :]
        gt_tmp = gt[b, 0, :, :]
        pred_tmp = pred[b, 0, :, :]

        rgb_tmp = rgb_tmp
        dep_tmp = dep_tmp / args.max_depth
        gt_tmp = gt_tmp / args.max_depth
        pred_tmp = pred_tmp / args.max_depth

        dep_tmp = (255.0 * cm(dep_tmp)).astype('uint8')
        gt_tmp = (255.0 * cm(gt_tmp)).astype('uint8')
        pred_tmp = (255.0 * cm(pred_tmp)).astype('uint8')

        rgb_tmp = 255.0 * np.transpose(rgb_tmp, (1, 2, 0))
        rgb_tmp = np.clip(rgb_tmp, 0, 256).astype('uint8')
        rgb_tmp = Image.fromarray(rgb_tmp, 'RGB')
        dep_tmp = Image.fromarray(dep_tmp[:, :, :3], 'RGB')
        gt_tmp = Image.fromarray(gt_tmp[:, :, :3], 'RGB')
        pred_tmp = Image.fromarray(pred_tmp[:, :, :3], 'RGB')

        # FIXME
        list_imgv = [rgb_tmp,
                        dep_tmp,
                        gt_tmp,
                        pred_tmp]

        widths, heights = zip(*(i.size for i in list_imgv))
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in list_imgv:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        list_imgh.append(new_im)

    widths, heights = zip(*(i.size for i in list_imgh))
    total_width = sum(widths)
    max_height = max(heights)
    img_total = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in list_imgh:
        img_total.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    path_output = f"output/{os.path.splitext(os.path.basename(args.model_path))[0]}"
    os.makedirs("output", exist_ok=True)
    os.makedirs(path_output, exist_ok=True)
    if not args.test:
        path_save = '{}/epoch{:04d}_{:08d}_result.png'.format(path_output, epoch, idx)
    else:
        os.makedirs('{}/depth_analy'.format(path_output), exist_ok=True)
        os.makedirs('{}/depth_rgb'.format(path_output), exist_ok=True)
        path_save = '{}/depth_analy/{}'.format(path_output, '{}.jpg'.format(idx))
        pred_tmp.save('{}/depth_rgb/{}'.format(path_output, '{}.jpg'.format(idx)))
    img_total.save(path_save)


def test(args):

    # DATASET
    print("Preparing data...")
    global sample_, output_, metric_txt_dir
    data_test = KittiDepth(args.test_option, args)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    print('Done!')
    # LOAD MODEL
    print("Prepare model...")
    core = ov.Core()
    ov_model = core.read_model(args.model_path)
    if args.device == 'GPU':
        INPUT_SHAPE_DICT = {
            "depth": [1, 1, 256, 1216],
            "img": [1, 3, 256, 1216],
            "lidar": [1, 1, 256, 1216],
            "d_clear": [1, 1, 256, 1216]
        }
        ov_model.reshape(INPUT_SHAPE_DICT)    
    ov_cmodel = ov.compile_model(ov_model, device_name=args.device)
    print('Done!')


    num_sample = len(loader_test) * loader_test.batch_size
    pbar_ = tqdm(total=num_sample)
    t_total = 0
    for batch_, sample_ in enumerate(loader_test):

        t0 = time.time()

        samplep = {key: val.float() for key, val in sample_.items()
                    if torch.is_tensor(val)}
        samplep['d_path'] = sample_['d_path']

        ov_output = ov_cmodel([samplep['dep'], samplep['rgb'], samplep['ip'], samplep['dep_clear']])
        t1 = time.time()
        t_total += (t1 - t0)
        
        save_result(args.epochs, batch_, samplep, ov_output, args)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        pbar_.set_description(error_str)
        pbar_.update(loader_test.batch_size)

    pbar_.close()
    t_avg = t_total / num_sample

    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


if __name__ == '__main__':
    test(config)
