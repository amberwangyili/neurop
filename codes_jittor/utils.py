 # Modified From https://github.com/hejingwenhejingwen/CSRNet/blob/master/codes/options/options.py
import os
from collections import OrderedDict
import logging
import yaml
import random
import cv2
from shutil import get_terminal_size
import math
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from datetime import datetime
import time
import sys
import numpy as np
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab, deltaE_cie76

def calculate_dE(img1, img2):
    return np.array(deltaE_cie76(rgb2lab(img1),rgb2lab(img2))).mean()

def calculate_ssim(img1, img2):
    return structural_similarity(img1,img2,data_range=1.0, channel_axis=-1,multichannel=True)
    

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return -20 * math.log10(math.sqrt(mse))

def aug_process(img_GT, img_LQ, img_M=None):
    h, w = img_GT.shape[:2]
    crop_size = 20
    new_h = random.randint(h - crop_size, h - 1)
    new_w = random.randint(w - crop_size, w - 1)

    y = random.randint(0, h - new_h - 1)
    x = random.randint(0, w - new_w - 1)
    img_GT = img_GT[y:y+new_h, x:x+new_w,:]
    img_LQ = img_LQ[y:y+new_h, x:x+new_w,:]
    if img_M is not None:
        img_M = img_M[y:y+new_h, x:x+new_w]

    is_flip = random.randint(0,3)
    if is_flip == 0:
        img_GT = np.flip(img_GT, axis=0)
        img_LQ = np.flip(img_LQ, axis=0)
        if img_M is not None:
            img_M = np.flip(img_M,axis=0)
    elif is_flip == 2:
        img_GT = np.flip(img_GT, axis=1)
        img_LQ = np.flip(img_LQ, axis=1)
        if img_M is not None:
            img_M = np.flip(img_M, axis=1)
    is_rot = random.randint(0,3)
    if is_rot !=0:
        if img_M is not None:
            img_M = np.rot90(img_M, is_rot)
        img_GT = np.rot90(img_GT, is_rot)
        img_LQ = np.rot90(img_LQ, is_rot)
    if img_M is not None:
        return img_GT, img_LQ, img_M
    else:
        return img_GT, img_LQ


def get_file_paths(folder,suffix):
    file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.split('.')[-1] == suffix:
                file_paths.append(file_path)
        break  
    return file_paths

def get_file_name(fp):
    return fp.split('/')[-1].split('.')[0]

    
class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

def get_timestamp():
    return datetime.now().strftime('_%y%m%d-%H%M%S')

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper
def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def parse(opt_path):
    Loader, _ = OrderedYaml()
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    
    # path
    opt['path'] = {}
    opt['path']['root'] = os.path.abspath(os.path.join(os.path.pardir))
    # datasets
    opt['datasets']['dataroot'] = os.path.join(opt['path']['root'], 'datasets', opt['datasets']['dataroot'])
    
    if opt['is_train']:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
        
    else:
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
    return opt