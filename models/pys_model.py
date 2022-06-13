import torch
from torch import nn
import torch.nn.functional as F


class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert  len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation,bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forwrd(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


class PyConv4(nn.Module):

    def __init__(self, inplanes, planes, pyconv_kernels=[3,5,7,9], stride=1, pyconv_groups=[1,4,8,16], k_size=3):
        super(PyConv4, self).__init__()

        self.conv2_1 = nn.Conv2d(inplanes, planes // 4, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0]//2, dilation=1, groups=pyconv_groups[0], bias=False)


        self.conv2_2 = nn.Conv2d(inplanes, planes // 4, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1] // 2, dilation=1, groups=pyconv_groups[1], bias=False)


        self.conv2_3 = nn.Conv2d(inplanes, planes // 4, kernel_size=pyconv_kernels[2], stride=stride,
                                 padding=pyconv_kernels[2] // 2, dilation=1, groups=pyconv_groups[2], bias=False)


        self.conv2_4 = nn.Conv2d(inplanes, planes // 4, kernel_size=pyconv_kernels[3], stride=stride,
                                 padding=pyconv_kernels[3] // 2, dilation=1, groups=pyconv_groups[3], bias=False)


    def forward(self, x):
        conv2_1 = self.conv2_1(x)
        conv2_2 = self.conv2_1(x)
        conv2_3 = self.conv2_1(x)
        conv2_4 = self.conv2_1(x)

        return torch.cat((conv2_1,conv2_2,conv2_3,conv2_4),dim=1)

class GlobalPyConvBlock(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(GlobalPyConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins),
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1,bias=False),
            BatchNorm(reduction_dim),
            nn.ReLU(inplace=True),
            PyConv4(reduction_dim,reduction_dim),
            BatchNorm(reduction_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_dim,reduction_dim,kernel_size=1, bias=False),
            BatchNorm(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        x = F.interpolate(self.features(x),x_size[2:],mode='bilinear',align_corners=True)
        return x

class LocalPyConvBlock(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(LocalPyConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            PyConv4(inplanes // reduction1, inplanes // reduction1),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class MergeLocalGlocal(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergeLocalGlocal, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x

class PyConvHead(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PyConvHead, self).__init__()
        out_size_local_context = 512
        out_size_global_context = 512
        self.local_context = LocalPyConvBlock(inplanes, out_size_local_context, BatchNorm, reduction1=4)
        self.global_context = GlobalPyConvBlock(inplanes, out_size_global_context, 9, BatchNorm)
        self.merge_context = MergeLocalGlocal(out_size_local_context + out_size_global_context, planes, BatchNorm)

    def forward(self, x):
        x = self.merge_context(self.local_context(x), self.global_context(x))
        return x








