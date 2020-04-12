from model import common
from model.layers import *
import torch
import torch.nn as nn


#### multiply
class Discriminator_face(nn.Module):
    def __init__(self, args):
        super(Discriminator_face,self).__init__()
        in_channels = args.n_colors
        out_channels = 64
        depth = 7
        conv=common.default_conv
        act = nn.LeakyReLU(True)

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))

        m_features.append(common.ResBlock(conv, out_channels, kernel_size=3, bn=True, act=act, res_scale=1))
        m_features.append(common.ResBlock(conv, out_channels, kernel_size=3, bn=True, act=act, res_scale=1))

        patch_size = args.patch_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x, masks):
        features = self.features(x * (masks[:,2,:,:].unsqueeze(1)))
        output = self.classifier(features.view(features.size(0), -1))

        return output


# #### concat
# class Discriminator_2(nn.Module):
#     def __init__(self, args):
#         super(Discriminator_2,self).__init__()
#         in_channels = args.n_colors + 10
#         out_channels = 64
#         depth = 7
#         conv=common.default_conv
#         act = nn.LeakyReLU(True)

#         def _block(_in_channels, _out_channels, stride=1):
#             return nn.Sequential(
#                 nn.Conv2d(
#                     _in_channels,
#                     _out_channels,
#                     3,
#                     padding=1,
#                     stride=stride,
#                     bias=False
#                 ),
#                 nn.BatchNorm2d(_out_channels),
#                 nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             )

#         m_features = [_block(in_channels, out_channels)]
#         for i in range(depth):
#             in_channels = out_channels
#             if i % 2 == 1:
#                 stride = 1
#                 out_channels *= 2
#             else:
#                 stride = 2
#             m_features.append(_block(in_channels, out_channels, stride=stride))

#         m_features.append(common.ResBlock(conv, out_channels, kernel_size=3, bn=True, act=act, res_scale=1))
#         m_features.append(common.ResBlock(conv, out_channels, kernel_size=3, bn=True, act=act, res_scale=1))

#         patch_size = args.patch_size // (2**((depth + 1) // 2))
#         m_classifier = [
#             nn.Linear(out_channels * patch_size**2, 1024),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Linear(1024, 1)
#         ]

#         self.features = nn.Sequential(*m_features)
#         self.classifier = nn.Sequential(*m_classifier)

#     def forward(self, x, masks):
#         features = self.features(torch.cat([x, masks], 1))
#         output = self.classifier(features.view(features.size(0), -1))

#         return output

