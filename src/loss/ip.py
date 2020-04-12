from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, utils

from model.light_cnn import LightCNN_29Layers_v2

import numpy as np
import cv2

class IP(nn.Module):
    def __init__(self, args):
        super(IP, self).__init__()
        self.model_recognition = LightCNN_29Layers_v2(num_classes=346)
        self.model_recognition = torch.nn.DataParallel(self.model_recognition).cuda()
        checkpoint = torch.load("lightCNN_pretrain.pth.tar")
        self.model_recognition.load_state_dict(checkpoint['state_dict'])

        self.submean = common.MeanShift(args.rgb_range)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.submean(x)
            x = self.model_recognition(x)
            return x

        out_sr, feat_sr = _forward(sr[0])
        with torch.no_grad():
            out_hr, feat_hr = _forward(hr[0].detach())

        loss = F.mse_loss(feat_sr, feat_hr) + F.mse_loss(out_sr, out_hr) 
        return loss
