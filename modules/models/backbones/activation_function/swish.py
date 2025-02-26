import torch
import torch.nn as nn


'''Swish'''
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    '''forward'''
    def forward(self, x):
        return x * nn.Sigmoid(x)