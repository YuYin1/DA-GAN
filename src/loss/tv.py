import utility
from types import SimpleNamespace

from model import common
import torch
import torch.nn as nn

class TV(nn.Module):
    def __init__(self, args):
        super(TV, self).__init__()
        #total_variation


    def forward(self, outputs, targets):
        fake = outputs[0]
        tv_loss = torch.mean(torch.abs(fake[:,:,:-1,:] - fake[:,:,1:,:])) + torch.mean(torch.abs(fake[:,:,:,:-1] - fake[:,:,:,1:]))  

        return tv_loss