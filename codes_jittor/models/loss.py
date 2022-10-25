from jittor import Module
import jittor as jt
import jittor.nn as nn

class TVLoss(Module):
    def __init__(self,TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight
    def execute(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = jt.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        w_tv = jt.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class CosineLoss(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x, y):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view(-1, 1, c)
        y = y.permute(0, 2, 3, 1).view(-1, 1, c)
        x = self.normalize(x)
        y = self.normalize(y)
        cos = jt.bmm_transpose(x,y)
        loss = 1-jt.sum(cos) / (1.0 * b * h * w)
        return loss    
    def normalize(self,x, axis=-1):
        x = 1. * x / (jt.norm(x, p=2, dim=axis, keepdim=True).expand_as(x) + 1e-12)
        return x