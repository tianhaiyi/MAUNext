'''
author:YUhang Wang
Guizhou University
'''
import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math

#  the coordinate attention
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=4):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out =  s_h.expand_as(x) * s_w.expand_as(x)

        return out

# the CSF model
class CSF(nn.Module):
    def __init__(self, in_channels,h,w):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.ca = CA_Block(in_channels, h,w)

    def forward(self, x, skip_x):
        skip_x_att = torch.add(x, skip_x)
        skip_x_att = self.ca(skip_x_att)
        skip_x_att = torch.mul(skip_x, skip_x_att)
        x = torch.add(skip_x_att, x)  # dim 1 is the channel dimension
        return x


# the MAC and FCF model
class FCF(nn.Module):
    def __init__(self, hws,channels=64, r=4):
        super(FCF, self).__init__()
        inter_channels = int(channels // r)

        # self.local_att = nn.Sequential(
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )
        #
        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )
        self.hws = hws
        self.ca = CA_Block(channels, h=hws, w=hws)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        # xl = self.local_att(xa)
        # xg = self.global_att(xa)
        # xlg = xl + xg
        # wei = self.sigmoid(xlg)
        wei = self.ca(xa)
        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)
        xi = x1+x2
        wei2 = self.ca(xi)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, hws, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 4, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        # self.ca = CA_Block(planes // 4, h=hws, w=hws)
        # self.se = SEWeightModule(planes // 4, h=hws, w=hws)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)
        self.hws = hws
        self.fusion = FCF(hws, planes)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        out = self.fusion(x1, x2)
        out = self.fusion(out, x3)
        out = self.fusion(out, x4)



        # feats = torch.cat((x1, x2, x3, x4), dim=1)
        # feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        #
        # x1_se = self.ca(x1)
        # x2_se = self.ca(x2)
        # x3_se = self.ca(x3)
        # x4_se = self.ca(x4)
        #
        # x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # attention_vectors = x_se.view(batch_size, 4, self.split_channel, self.hws, self.hws)
        # attention_vectors = self.softmax(attention_vectors)
        # feats_weight = feats * attention_vectors
        # for i in range(4):
        #     x_se_weight_fp = feats_weight[:, i, :, :]
        #     if i == 0:
        #         out = x_se_weight_fp
        #     else:
        #         out = torch.cat((x_se_weight_fp, out), 1)

        return out


class MACBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, hws, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(MACBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, hws, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)



# the NMLP model
class mixmlp_c(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_c(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(9, 1)
        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        nn_Unfold = nn.Unfold(kernel_size=[3, 3], dilation=1, padding=1, stride=1)
        x1 = nn_Unfold(xn)
        x2 = x1.view(B, C, 9, N).contiguous()
        x2 = x2.transpose(2, 3)
        x3 = self.fc3(x2)
        x_s = x3.reshape(B, C, H * W).contiguous()

        x_s = x_s.transpose(1, 2)

        x = self.fc1(x_s)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)

        return x

class mixmlp_s(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features =  in_features
        hidden_features =  in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_s(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(9, 1)
        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape
        C_ = int(math.sqrt(C))
        xn = x.view(B, N, C_, C_).contiguous()
        nn_Unfold = nn.Unfold(kernel_size=[3, 3], dilation=1, padding=1, stride=1)
        x1 = nn_Unfold(xn)
        x2 = x1.view(B, N, 9, C).contiguous()
        x2 = x2.transpose(2, 3)
        x3 = self.fc3(x2)

        x_s = x3.reshape(B, N, C).transpose(1, 2).contiguous()
        x = self.fc1(x_s)

        x = self.dwconv(x, C_)
        x = self.act(x)
        x = self.fc2(x)
        x = x.transpose(1, 2)
        return x

class mixBlock(nn.Module):
    def __init__(self, dim,dim_2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_c = mixmlp_c(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.mlp_s = mixmlp_s(in_features=dim_2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp_c(self.norm2(x), H, W))
        x = x + self.drop_path(self.mlp_s(self.norm2(x), H, W))
        return x

class DWConv_c(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_c, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class DWConv_s(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_s, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, C_):
        B, C, N = x.shape
        x = x.transpose(1, 2).view(B, N, C_, C_)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x