from jittor import Module
import jittor as jt 
import jittor.nn as nn
import numpy as np
import torch


class Operator(Module):
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(Operator,self).__init__()
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1) 
        self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1) 
        self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        

    def execute(self, x, val):
        x_code = self.encoder(x)
        y_code = x_code + val
        y_code = nn.leaky_relu(self.mid_conv(y_code))
        y = self.decoder(y_code)
        return y


class Renderer(Module):
    
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(Renderer,self).__init__()
        self.in_nc = in_nc
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.ex_block = Operator(in_nc,out_nc,base_nf)
        self.bc_block = Operator(in_nc,out_nc,base_nf)
        self.vb_block = Operator(in_nc,out_nc,base_nf)
    
    def execute(self,x_ex,x_bc,x_vb,v_ex,v_bc,v_vb):
        
        rec_ex = self.ex_block(x_ex,0)
        rec_bc = self.bc_block(x_bc,0)
        rec_vb = self.vb_block(x_vb,0)

        map_ex = self.ex_block(x_ex,v_ex)
        map_bc = self.bc_block(x_bc,v_bc)
        map_vb = self.vb_block(x_vb,v_vb)

        return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb


    

class Encoder(Module):
    def __init__(self, in_nc=3, encode_nf=32):
        super(Encoder, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, encode_nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(encode_nf, encode_nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU()
        self.max = nn.AdaptiveMaxPool2d((1,1))

    def execute(self, x):
        b, _,_,_ = x.size()
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        std = jt.var(conv2_out, dims=(2, 3)).sqrt()
        mean = jt.mean(conv2_out, dims=(2, 3))
        maxs = self.max(conv2_out).squeeze(2).squeeze(2)
        out = jt.concat([std, mean, maxs], dim=1)
        return out


class Predictor(Module):
    def __init__(self,fea_dim):
        super(Predictor,self).__init__()
        self.fc3 = nn.Linear(fea_dim,1)
        self.tanh = nn.Tanh()
    def execute(self,img_fea):
        val = self.tanh(self.fc3(img_fea))
        return val    

    
class NeurOP(Module):
    def __init__(self, in_nc=3, out_nc = 3, base_nf = 64, encode_nf =32 , load_path = None):
        super(NeurOP,self).__init__()
        self.fea_dim = encode_nf * 3
        self.image_encoder = Encoder(in_nc,encode_nf)
        renderer = Renderer(in_nc,out_nc,base_nf)
        if load_path is not None: 
            renderer.load(load_path)
            
        self.bc_renderer = renderer.bc_block
        self.bc_predictor =  Predictor(self.fea_dim)
        
        self.ex_renderer = renderer.ex_block
        self.ex_predictor =  Predictor(self.fea_dim)
        
        self.vb_renderer = renderer.vb_block
        self.vb_predictor =  Predictor(self.fea_dim)

        self.renderers = [self.bc_renderer,self.ex_renderer,self.vb_renderer]
        self.predict_heads = [self.bc_predictor,self.ex_predictor,self.vb_predictor]
            
    def render(self,x,vals):
        b,_,h,w = img.shape
        imgs = []
        for nop, scalar in zip(self.renderers,vals):
            img = nop(img,scalar)
            output_img = jt.clamp(img, 0, 1.0)
            imgs.append(output_img)
        return imgs
    
    def execute(self,img, return_vals = True):
        b,_,h,w = img.shape
        vals = []
        for nop, predict_head in zip(self.renderers,self.predict_heads):
            img_resized = nn.interpolate(img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
            feat = self.image_encoder(img_resized)
            scalar = predict_head(feat)
            vals.append(scalar)
            img = nop(img,scalar)
        img = jt.clamp(img, 0, 1.0)
        if return_vals:
            return img,vals
        else:
            return img