from utils import *
from data import build_train_loader, build_val_loader
import argparse
import logging
import os
import random
import numpy as np
import torch
from models import build_model
from tqdm import tqdm
from collections import defaultdict
import imageio
import torch
import jittor as jt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='path of the yaml configuration file')
    parser.add_argument('--save', action='store_true', help='if specified, save figures in result folder')

    args = parser.parse_args()
    opt = parse(args.config)
    opt = dict_to_nonedict(opt)

    for fp in opt['path']:
        path = (opt['path'][fp])
        if not os.path.exists(path):
            os.makedirs(path)
    logger_name = f"{opt['name']}{get_timestamp()}"
    logger = logging.getLogger('base')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)        
    log_file = f"{opt['path']['log']}/{logger_name}.log"    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info(dict2str(opt))


    dataset_opt = opt['datasets']
    val_loader = build_val_loader(dataset_opt)
    model = build_model(opt)

    #### inference
    psnr_list = []
    ssim_list = []
    deltaE_list = []

    pbar = ProgressBar(len(val_loader))
    jt.sync_all()
    model.netG.eval()
    for val_data in val_loader:
        img_name = get_file_name(val_data['LQ_path'][0])
        if args.save:
            img_dir = os.path.join(opt['path']['results_root'],'images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
        model.feed_data(val_data)
        visuals = model.test()
    
        sr_img = visuals['rlt']
        gt_img = visuals['GT']
        psnr = calculate_psnr(sr_img, gt_img)
        ssim = calculate_ssim(sr_img, gt_img)
        dE = calculate_dE(sr_img, gt_img)
        
    
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        deltaE_list.append(dE)
        pbar.update('img: {}  PSNR: {}  SSIM: {}  deltaE: {}'.format(img_name, psnr, ssim, dE))
        if args.save:
           save_img_path = os.path.join(img_dir,'{:s}.png'.format(img_name))
           imageio.imwrite(save_img_path, (255.0 * sr_img).astype('uint8'))
    jt.sync_all()    
    logger.info('Average PSNR: {}  SSIM: {}  deltaE: {}  Total image: {}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(deltaE_list), len(psnr_list)))
    
