#-*- coding:utf-8 -*-
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def projection_transH(original,norm):
    return original-np.sum(original*norm,axis=1,keepdims=True)*norm

