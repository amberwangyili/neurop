from models.networks import *
from models.loss import *

from torch.nn.parallel import DataParallel
import os
import logging
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn



class InitModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(opt['device'])
        self.name = "neurop_initialization"
        net_opt = opt['network_G']
        self.netG = Renderer(net_opt['in_nc'],net_opt['out_nc'],net_opt['base_nf']).to(self.device)
        self.netG = DataParallel(self.netG)
        self.print_network()
        self.load() 
        self.netG.train()

        train_opt = opt['train']
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=train_opt['lr_G'], weight_decay=0, betas=(train_opt['beta1'], train_opt['beta2']))
        self.log_dict = OrderedDict()

    def feed_data(self, data):

        self.A_ex = data['A_ex'].to(self.device) 
        self.A_bc = data['A_bc'].to(self.device) 
        self.A_vb = data['A_vb'].to(self.device) 

        self.B_ex = data['B_ex'].to(self.device) 
        self.B_bc = data['B_bc'].to(self.device) 
        self.B_vb = data['B_vb'].to(self.device) 

        self.val_ex = data['val_ex'].to(self.device) 
        self.val_bc = data['val_bc'].to(self.device) 
        self.val_vb = data['val_vb'].to(self.device) 
        
        self.rec_A_ex = None
        self.rec_A_bc = None
        self.rec_A_vb = None
        self.map_B_ex = None
        self.map_B_bc = None
        self.map_B_vb = None

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.rec_A_ex, self.rec_A_bc, self.rec_A_vb,self.map_B_ex, self.map_B_bc, self.map_B_vb = self.netG(self.A_ex,self.A_bc,self.A_vb,self.val_ex,self.val_bc,self.val_vb)

        loss_unary_ex = self.criterion(self.rec_A_ex, self.A_ex)
        loss_unary_bc = self.criterion(self.rec_A_bc, self.A_bc)
        loss_unary_vb = self.criterion(self.rec_A_vb, self.A_vb)
        loss_pair_ex = self.criterion(self.map_B_ex, self.B_ex)
        loss_pair_bc = self.criterion(self.map_B_bc, self.B_bc)
        loss_pair_vb = self.criterion(self.map_B_vb, self.B_vb)

        loss_unary = loss_unary_ex + loss_unary_bc + loss_unary_vb
        loss_pair = loss_pair_ex + loss_pair_bc + loss_pair_vb
        loss = loss_unary + loss_pair

        loss.backward()

        self.log_dict['loss_ex'] = loss_unary_ex.item() + loss_pair_ex.item()
        self.log_dict['loss_bc'] = loss_unary_bc.item() + loss_pair_bc.item()
        self.log_dict['loss_vb'] = loss_unary_vb.item() + loss_pair_vb.item()

        self.log_dict['loss_unary'] = loss_unary.item()
        self.log_dict['loss_pair'] = loss_pair.item()
        self.log_dict['loss_total'] = loss.item()
        self.optimizer.step()    

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.rec_A_ex, self.rec_A_bc, self.rec_A_vb,self.map_B_ex, self.map_B_bc, self.map_B_vb = self.netG(self.A_ex,self.A_bc,self.A_vb,self.val_ex,self.val_bc,self.val_vb)
        self.netG.train()        

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['map_B_ex'] = self.map_B_ex.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['map_B_bc'] = self.map_B_bc.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['map_B_vb'] = self.map_B_vb.detach()[0].float().cpu().numpy().transpose(1,2,0)

        out_dict['B_ex'] = self.B_ex.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['B_bc'] = self.B_bc.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['B_vb'] = self.B_vb.detach()[0].float().cpu().numpy().transpose(1,2,0)


        out_dict['rec_A_ex'] = self.rec_A_ex.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['rec_A_bc'] = self.rec_A_bc.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['rec_A_vb'] = self.rec_A_vb.detach()[0].float().cpu().numpy().transpose(1,2,0)

        out_dict['A_ex'] = self.A_ex.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['A_bc'] = self.A_bc.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['A_vb'] = self.A_vb.detach()[0].float().cpu().numpy().transpose(1,2,0)
        
        return out_dict



    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger = logging.getLogger('base')
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)



    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save(self, iter_label):
        network = self.netG
        save_filename = '{}_{}.pth'.format(iter_label, self.name)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        strict =  self.opt['path']['strict_load']
        network = self.netG

        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))

            if isinstance(network, nn.DataParallel):
                network = network.module
            load_net = torch.load(load_path)
            load_net_clean = OrderedDict()  # remove unnecessary 'module.'
            for k, v in load_net.items():
                if k.startswith('module.'):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            network.load_state_dict(load_net_clean, strict=strict)

            



class FinetuneModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(opt['device'])
        self.name = "neurop_"+opt['datasets']['name']
        net_opt = opt['network_G']
        self.netG = NeurOP(net_opt['in_nc'],net_opt['out_nc'],net_opt['base_nf'],net_opt['cond_nf'],net_opt['init_model'])
        if opt['weights'] is not None:
            self.netG.load_state_dict(torch.load(opt['weights']))
        self.netG.to(self.device)        
        self.netG = DataParallel(self.netG)
        self.print_network()
        if opt['is_train']:
            self.netG.train()
            train_opt = opt['train']
            self.cri_pix = nn.L1Loss().to(self.device)
            self.cri_cos = CosineLoss().to(self.device)
            self.cri_tv = TVLoss().to(self.device)
            self.cri_ratio = 1/train_opt['pixel_weight']
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=train_opt['lr_G'], weight_decay=0, betas=(train_opt['beta1'], train_opt['beta2']))
        self.log_dict = defaultdict(int)

    def feed_data(self, data):

        self.var_L = data['LQ'].to(self.device) 
        self.real_H = data['GT'].to(self.device)         
        self.fake_H = None

        
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.fake_H = self.netG(img=self.var_L,return_vals=False)
        loss_pix = self.cri_pix(self.fake_H, self.real_H)
        loss_cos = self.cri_cos(self.fake_H, self.real_H)
        loss_tv = self.cri_tv(self.fake_H)

        loss = loss_pix + self.cri_ratio * (loss_cos + loss_tv)
        loss.backward()

        self.log_dict['loss_pix'] += loss_pix.item()
        self.log_dict['loss_cos'] += loss_cos.item() 
        self.log_dict['loss_tv'] += loss_tv.item()
        self.log_dict['loss_total'] += loss.item()
        self.optimizer.step()    

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(img=self.var_L,return_vals=False)
        self.netG.train()        

    def get_current_log(self):
        return self.log_dict
    
    

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu().numpy().transpose(1,2,0)
        out_dict['GT'] = self.real_H.detach()[0].float().cpu().numpy().transpose(1,2,0)
        return out_dict



    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger = logging.getLogger('base')
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)



    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save(self, iter_label):
        network = self.netG
        save_filename = '{}_{}.pth'.format(iter_label, self.name)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
