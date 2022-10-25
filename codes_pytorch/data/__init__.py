"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data
from .dataset import *



def build_train_loader(dataset_opt):
    mode = dataset_opt['mode']
    num_workers = dataset_opt['n_cpus']
    batch_size = dataset_opt['batch_size']

    assert(batch_size == 1)

    if mode == 'init':
        dataset = InitDataset(dataset_opt) 
    elif mode == "finetune":
        dataset = FinetuneDataset(dataset_opt,phase="train")
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    
    logger = logging.getLogger('base')
    logger.info('Training Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, sampler=None, drop_last=False,
                                       pin_memory=False)


def build_val_loader(dataset_opt):
    mode = dataset_opt['mode']
    num_workers = dataset_opt['n_cpus']
    batch_size = dataset_opt['batch_size']

    assert(batch_size == 1)

    if mode == 'init':
        dataset = InitDataset(dataset_opt) 
    elif mode == "finetune":
        dataset = FinetuneDataset(dataset_opt,phase="val")
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    
    logger = logging.getLogger('base')
    logger.info('Validation Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                       num_workers=1, sampler=None, drop_last=False,
                                       pin_memory=False)

