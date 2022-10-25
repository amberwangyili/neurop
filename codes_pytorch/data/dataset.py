from utils import *
from collections import defaultdict
import imageio
import torch
from torch.utils.data import Dataset
import numpy as np


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
        
        val_ex = torch.tensor((int(get_file_name(B_ex).split('-')[-1]) - int(get_file_name(A_ex).split('-')[-1]))/20).float()
        val_bc = torch.tensor((int(get_file_name(B_bc).split('-')[-1]) - int(get_file_name(A_bc).split('-')[-1]))/20).float()
        val_vb = torch.tensor((int(get_file_name(B_vb).split('-')[-1]) - int(get_file_name(A_vb).split('-')[-1]))/20).float()


        img_A_ex = np.array(imageio.imread(A_ex))/255
        img_A_ex = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_ex, (2, 0, 1)))).float()
        img_B_ex = np.array(imageio.imread(B_ex))/255
        img_B_ex = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_ex, (2, 0, 1)))).float()

        img_A_bc = np.array(imageio.imread(A_bc))/255
        img_A_bc = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_bc, (2, 0, 1)))).float()
        img_B_bc = np.array(imageio.imread(B_bc))/255
        img_B_bc = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_bc, (2, 0, 1)))).float()

        img_A_vb = np.array(imageio.imread(A_vb))/255
        img_A_vb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_vb, (2, 0, 1)))).float()
        img_B_vb = np.array(imageio.imread(B_vb))/255
        img_B_vb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_vb, (2, 0, 1)))).float()

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
                img_A = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A, (2, 0, 1)))).float()
                img_B = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B, (2, 0, 1)))).float()       
                img_M = torch.from_numpy(np.ascontiguousarray(img_M)).long()   
                return {'idx': idx, 'LQ_path':path_A,'GT_path':path_B, 'LQ': img_A, 'GT': img_B, 'M': img_M}
            else:
                img_A, img_B = aug_process(img_A,img_B)
                img_A = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A, (2, 0, 1)))).float()
                img_B = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B, (2, 0, 1)))).float()       
                return {'idx': idx, 'LQ_path':path_A,'GT_path':path_B, 'LQ': img_A, 'GT': img_B}
        else:
            img_A = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A, (2, 0, 1)))).float()
            img_B = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B, (2, 0, 1)))).float()       
            return {'idx': idx, 'LQ_path':path_A,'GT_path':path_B, 'LQ': img_A, 'GT': img_B}

    def __len__(self):
        return len(self.path_A)
