#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class marginLoss(nn.Module):
    def __init__(self):
        super(marginLoss,self).__init__()

    def forward(self,pos,neg,margin):
        zero_tensor=torch.FloatTensor(pos.size())
        margin=Variable(torch.FloatTensor([margin]))
        zero_tensor.zero_()
        zero_tensor=Variable(zero_tensor)
        return torch.sum(torch.max(pos-neg+margin,zero_tensor))
