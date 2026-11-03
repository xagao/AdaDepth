from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from hr_layers import *
from layers import upsample

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

## ChannelAttetion
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature

## SpatialAttetion

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, in_feature):
        x = in_feature
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #x = avg_out
        #x = max_out
        x = self.conv1(x)
        return self.sigmoid(x).expand_as(in_feature) * in_feature


#CS means channel-spatial  
class CS_Block(nn.Module):
    def __init__(self, in_channel, reduction = 16 ):
        super(CS_Block, self).__init__()
        
        reduction = reduction
        in_channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channel // reduction, in_channel, bias = False)
            )
        self.sigmoid = nn.Sigmoid()
        ## Spatial_Block
        self.conv = nn.Conv2d(2,1,kernel_size = 1, bias = False)
        #self.conv = nn.Conv2d(1,1,kernel_size = 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, in_feature):

        b,c,_,_ = in_feature.size()
        
        
        output_weights_avg = self.avg_pool(in_feature).view(b,c)
        output_weights_max = self.max_pool(in_feature).view(b,c)
         
        output_weights_avg = self.fc(output_weights_avg).view(b,c,1,1)
        output_weights_max = self.fc(output_weights_max).view(b,c,1,1)
        
        output_weights = output_weights_avg + output_weights_max
        
        output_weights = self.sigmoid(output_weights)
        out_feature_1 = output_weights.expand_as(in_feature) * in_feature
        
        ## Spatial_Block
        in_feature_avg = torch.mean(out_feature_1,1,True)
        in_feature_max,_ = torch.max(out_feature_1,1,True)
        mixed_feature = torch.cat([in_feature_avg,in_feature_max],1)
        spatial_attention = self.sigmoid(self.conv(mixed_feature))
        out_feature = spatial_attention.expand_as(out_feature_1) * out_feature_1
        #########################
        
        return out_feature
        
class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel = None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        #self.sa = SpatialAttention()
        #self.cs = CS_Block(channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1 )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, high_features, low_features):

        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        features = self.ca(features)
        #features = self.sa(features)
        #features = self.cs(features)
        
        return self.relu(self.conv_se(features))

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
        
        # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else: 
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
            self.convs["72"] = Attention_Module(self.num_ch_enc[4]  , self.num_ch_enc[3] * 2, 256)
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128)
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64 , 64)
            self.convs["9"] = Attention_Module(64, 64, 32)
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72 , feature36)
        x18 = self.convs["18"](x36 , feature18)
        x9 = self.convs["9"](x18,[feature64])
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))
        
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        return outputs
        
