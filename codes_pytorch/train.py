from utils import *
from data import build_train_loader,build_val_loader
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./configs/init_neurop.yaml')
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

    model = build_model(opt)
    dataset_opt = opt['datasets']
    train_loader = build_train_loader(dataset_opt)
    current_step = 0
    start_epoch = 0
    seed = opt['train']['manual_seed']
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_iters = opt['train']['niter']
    total_epochs = int(total_iters/len(train_loader))

    for epoch in tqdm(range(start_epoch, total_epochs + 1)):
        for _, train_data in enumerate((train_loader)):
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            model.feed_data(train_data)
            model.optimize_parameters()
        #### log
        logs = model.get_current_log()
        message = '[epoch:{:3d}, iter:{:8,d}, '.format(epoch, current_step)
        for k,v in logs.items():
            v /= len(train_loader)
            message += '{:s}: {:.4e} '.format(k, v)
        model.log_dict = defaultdict(int)
        logger.info(message)
        model.save("latest")
