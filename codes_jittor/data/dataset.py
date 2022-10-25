from collections import defaultdict
import numpy as np
import os
import numpy as np
import imageio
import random

import jittor as jt 
import jittor.transform as transform
from jittor.dataset.dataset import Dataset

import math
import itertools


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

def aug_process(img_GT, img_LQ):
    h, w = img_GT.shape[:2]
    crop_size = 20
    new_h = random.randint(h - crop_size, h - 1)
    new_w = random.randint(w - crop_size, w - 1)

    y = random.randint(0, h - new_h - 1)
    x = random.randint(0, w - new_w - 1)
    img_GT = img_GT[y:y+new_h, x:x+new_w,:]
    img_LQ = img_LQ[y:y+new_h, x:x+new_w,:]
    is_flip = random.randint(0,3)
    if is_flip == 0:
        img_GT = np.flip(img_GT, axis=0)
        img_LQ = np.flip(img_LQ, axis=0)
    elif is_flip == 2:
        img_GT = np.flip(img_GT, axis=1)
        img_LQ = np.flip(img_LQ, axis=1)
    is_rot = random.randint(0,3)
    if is_rot !=0:
        img_GT = np.rot90(img_GT, is_rot)
        img_LQ = np.rot90(img_LQ, is_rot)
    return img_GT, img_LQ





class InitDataset(Dataset):
    def __init__(self,dataset_opt):
        super(InitDataset,self).__init__()
        opt = dataset_opt
        filepath_EX = get_file_paths(os.path.join(opt['dataroot'],'EX'),'png')
        filepath_BC = get_file_paths(os.path.join(opt['dataroot'],'BC'),'png')
        filepath_VB = get_file_paths(os.path.join(opt['dataroot'],'VB'),'png')


        self.file_ex = defaultdict(list)
        self.file_bc = defaultdict(list)
        self.file_vb = defaultdict(list)

        for f_ex,f_bc,f_vb in zip(filepath_EX,filepath_BC,filepath_VB):
            idx_ex = f_ex.split('/')[-1].split('-')[0]
            idx_bc = f_bc.split('/')[-1].split('-')[0]
            idx_vb = f_vb.split('/')[-1].split('-')[0]
            self.file_ex[idx_ex].append(f_ex)
            self.file_bc[idx_bc].append(f_bc)
            self.file_vb[idx_vb].append(f_vb)

        self.file_keys = list(self.file_ex.keys())
    def __len__(self):
        return len(self.file_keys)
    def __getitem__(self,index):       
        key = self.file_keys[index]    
        A_ex, B_ex = np.random.choice(self.file_ex[key],2,replace=False)
        A_bc, B_bc = np.random.choice(self.file_bc[key],2,replace=False)
        A_vb, B_vb = np.random.choice(self.file_vb[key],2,replace=False)
        

        val_ex = jt.float((int(get_file_name(B_ex).split('-')[-1]) - int(get_file_name(A_ex).split('-')[-1]))/20)
        val_bc = jt.float((int(get_file_name(B_bc).split('-')[-1]) - int(get_file_name(A_bc).split('-')[-1]))/20)
        val_vb = jt.float((int(get_file_name(B_vb).split('-')[-1]) - int(get_file_name(A_vb).split('-')[-1]))/20)


        img_A_ex = np.array(imageio.imread(A_ex))/255
        img_A_ex = jt.array(np.ascontiguousarray(np.transpose(img_A_ex, (2, 0, 1)))).float()
        img_B_ex = np.array(imageio.imread(B_ex))/255
        img_B_ex = jt.array(np.ascontiguousarray(np.transpose(img_B_ex, (2, 0, 1)))).float()

        img_A_bc = np.array(imageio.imread(A_bc))/255
        img_A_bc = jt.array(np.ascontiguousarray(np.transpose(img_A_bc, (2, 0, 1)))).float()
        img_B_bc = np.array(imageio.imread(B_bc))/255
        img_B_bc = jt.array(np.ascontiguousarray(np.transpose(img_B_bc, (2, 0, 1)))).float()

        img_A_vb = np.array(imageio.imread(A_vb))/255
        img_A_vb = jt.array(np.ascontiguousarray(np.transpose(img_A_vb, (2, 0, 1)))).float()
        img_B_vb = np.array(imageio.imread(B_vb))/255
        img_B_vb = jt.array(np.ascontiguousarray(np.transpose(img_B_vb, (2, 0, 1)))).float()

        return {'A_ex': img_A_ex, 'B_ex': img_B_ex, 'val_ex':val_ex, 
                'A_bc': img_A_bc, 'B_bc': img_B_bc, 'val_bc':val_bc, 
                'A_vb': img_A_vb, 'B_vb': img_B_vb, 'val_vb':val_vb 
               } 



class FinetuneDataset(Dataset):
    def __init__(self,opt,phase):
        super(FinetuneDataset, self).__init__()
        self.opt = opt
        self.phase = phase
        
        if self.opt['name'] == "fivek_dark":
            self.suffix_A = "tif"
            self.suffix_B = "jpg"
            self.dataroot_A = opt['dataroot']
            self.dataroot_B = opt['dataroot']

        elif self.opt['name'] == "fivek_lite":
            self.suffix_A = "jpg"
            self.suffix_B = "jpg"
            self.dataroot_A = opt['dataroot']
            self.dataroot_B = opt['dataroot']

        elif "ppr" in self.opt['name']:
            self.suffix_A = "tif"
            self.suffix_B = "tif"
            self.suffix_M = "png"
            self.dataroot_A = opt['dataroot']
            self.dataroot_B = os.path.join(opt['dataroot'],opt['name'])

        else:
            raise NotImplementedError('Datasetname [{:s}] is not recognized.'.format(self.opt['name']))

        if self.phase == "train":
            self.path_A = get_file_paths(os.path.join(self.dataroot_A,"trainA"),self.suffix_A)
            self.path_B = get_file_paths(os.path.join(self.dataroot_B,"trainB"),self.suffix_B)
            if "ppr" in self.opt['name']:
                self.path_M = get_file_paths(os.path.join(self.dataroot_A,"trainM"),self.suffix_M)

        else:
            self.path_A = get_file_paths(os.path.join(self.dataroot_A,"testA"),self.suffix_A)
            self.path_B = get_file_paths(os.path.join(self.dataroot_B,"testB"),self.suffix_B)
            if "ppr" in self.opt['name']:
                self.path_M = get_file_paths(os.path.join(self.dataroot_A,"testM"),self.suffix_M)

        assert(len(self.path_A) > 0)
        assert(len(self.path_A) == len(self.path_B))
        
    def __getitem__(self,index):        
        path_A,path_B = self.path_A[index],self.path_B[index]
        idx = get_file_name(path_A)
        
        if self.suffix_A == "tif":
            img_A = np.array(imageio.imread(path_A))/(2**16-1)
        else:
            img_A = np.array(imageio.imread(path_A))/(2**8-1)

        img_B = np.array(imageio.imread(path_B))/(2**8-1)
            
        if self.phase == "train":
            if "ppr" in self.opt['name']:
                path_M = self.path_M[index]
                img_M = np.array(imageio.imread(path_M))
                img_A, img_B, img_M = aug_process(img_A,img_B,img_M)
                return {'idx': idx, 'LQ_path':path_A,'GT_path':path_B, 'LQ': img_A, 'GT': img_B, 'M': img_M}
            else:
                img_A, img_B = aug_process(img_A,img_B)
                return {'idx': idx, 'LQ_path':path_A,'GT_path':path_B, 'LQ': img_A, 'GT': img_B}
        else:
            return {'idx': idx, 'LQ_path':path_A,'GT_path':path_B, 'LQ': img_A, 'GT': img_B}

    def __len__(self):
        return len(self.path_A)
