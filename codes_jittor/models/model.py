from models.networks import *
from models.loss import *
import os
import logging
from collections import OrderedDict, defaultdict
import torch


class InitModel():
    def __init__(self, opt):
        self.opt = opt
        self.name = "neurop_initialization"
        net_opt = opt['network_G']
        if opt['device'] == 'cuda':
            jt.flags.use_cuda = 1

        self.netG = Renderer(net_opt['in_nc'],net_opt['out_nc'],net_opt['base_nf'])
        self.netG.train()

        train_opt = opt['train']
        self.criterion = nn.L1Loss()
        self.optimizer = jt.optim.Adam(self.netG.parameters(), lr=train_opt['lr_G'], weight_decay=0, betas=(train_opt['beta1'], train_opt['beta2']))
        self.log_dict = OrderedDict()

    def feed_data(self, data):

        self.A_ex = data['A_ex']
        self.A_bc = data['A_bc']
        self.A_vb = data['A_vb']

        self.B_ex = data['B_ex']
        self.B_bc = data['B_bc']
        self.B_vb = data['B_vb']

        self.val_ex = data['val_ex']
        self.val_bc = data['val_bc']
        self.val_vb = data['val_vb']
        
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

        self.optimizer.step(loss)

        self.log_dict['loss_ex'] = loss_unary_ex.item() + loss_pair_ex.item()
        self.log_dict['loss_bc'] = loss_unary_bc.item() + loss_pair_bc.item()
        self.log_dict['loss_vb'] = loss_unary_vb.item() + loss_pair_vb.item()

        self.log_dict['loss_unary'] = loss_unary.item()
        self.log_dict['loss_pair'] = loss_pair.item()
        self.log_dict['loss_total'] = loss.item()

    def test(self):
        self.netG.eval()
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



    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    
    def save(self, iter_label):
        network = self.netG
        save_filename = '{}_{}.pkl'.format(iter_label, self.name)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        network.save(save_path)

            



class FinetuneModel():

    def __init__(self, opt):
        self.opt = opt
        self.name = "neurop_"+opt['datasets']['name']
        net_opt = opt['network_G']
        if opt['device'] == 'cuda':
            jt.flags.use_cuda = 1
        self.netG = NeurOP(net_opt['in_nc'],net_opt['out_nc'],net_opt['base_nf'],net_opt['cond_nf'],net_opt['init_model'])
        if opt['weights'] is not None:
            self.netG.load_state_dict(torch.load(opt['weights']))
        if opt['is_train']:
            self.netG.train()
            train_opt = opt['train']
            self.cri_pix = nn.L1Loss()
            self.cri_cos = CosineLoss()
            self.cri_tv = TVLoss()
            self.cri_ratio = 1/train_opt['pixel_weight']
            self.optimizer = jt.optim.Adam(self.netG.parameters(), lr=train_opt['lr_G'], weight_decay=0, betas=(train_opt['beta1'], train_opt['beta2']))
        self.log_dict = defaultdict(int)

    def feed_data(self, data):
        self.var_L = jt.array(np.ascontiguousarray(np.transpose(data['LQ'][0], (2, 0, 1)))).unsqueeze(0)
        self.real_H = jt.array(np.ascontiguousarray(np.transpose(data['GT'][0], (2, 0, 1)))).unsqueeze(0)
        self.fake_H = None

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.fake_H = self.netG(img=self.var_L,return_vals=False)
        loss_pix = self.cri_pix(self.fake_H, self.real_H)
        loss_cos = self.cri_cos(self.fake_H, self.real_H)
        loss_tv = self.cri_tv(self.fake_H)

        loss = loss_pix + self.cri_ratio * (loss_cos + loss_tv)
        
        self.log_dict['loss_pix'] += loss_pix.item()
        self.log_dict['loss_cos'] += loss_cos.item() 
        self.log_dict['loss_tv'] += loss_tv.item()
        self.log_dict['loss_total'] += loss.item()
        self.optimizer.step(loss)    

    def test(self):
        self.netG.eval()
        self.fake_H = self.netG(img=self.var_L,return_vals=False)
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.float().data[0].transpose(1,2,0)
        out_dict['rlt'] = self.fake_H.float().data[0].transpose(1,2,0)
        out_dict['GT'] = self.real_H.float().data[0].transpose(1,2,0)
        return out_dict 

    def get_current_log(self):
        return self.log_dict
    

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.float().data[0].transpose(1,2,0)
        out_dict['rlt'] = self.fake_H.float().data[0].transpose(1,2,0)
        out_dict['GT'] = self.real_H.float().data[0].transpose(1,2,0)
        return out_dict


    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def save(self, iter_label):
        network = self.netG
        save_filename = '{}_{}.pkl'.format(iter_label, self.name)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        network.save(save_path)

            



