import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime, timedelta
from decoder.dec_net import Decoder_Network
from encoder_dgcnn.dgcnn import DGCNN
from encoder_image.resnet import ResNet
from config import params

p = params()


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        # Encoders for images and Point clouds
        self.pc_encoder = DGCNN()
        self.im_encoder = ResNet()

        # Attention layers to fuse the information from the two modalities

        self.cross_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(p.d_attn)

        self.self_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(p.d_attn)

        self.cross_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(p.d_attn)

        self.self_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(p.d_attn)

        self.cross_attn3 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(p.d_attn)


        # Decoder Network to reconstruct the point cloud
        self.decoder = Decoder_Network()

    def forward(self, x_part, view):

       
        pc_feat = self.pc_encoder(x_part)  #B x F x N 
        im_feat = self.im_encoder(view)  #B x F x N

        im_feat = im_feat.permute(0, 2, 1)
        pc_feat = pc_feat.permute(0, 2, 1)

        x, _ = self.cross_attn1(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm1(x + pc_feat) # B x N x F
        

        x, _ = self.self_attn1(pc_feat, pc_feat, pc_feat)
        pc_feat = self.layer_norm2(x + pc_feat)
        pc_skip = pc_feat
        
        x, _ = self.cross_attn2(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm3(x + pc_feat)

        x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)
        pc_feat = self.layer_norm4(x + pc_feat)

        x, _ = self.cross_attn3(pc_feat, pc_skip, pc_skip)
        pc_feat = self.layer_norm5(x + pc_feat)

        x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3
        

        final = self.decoder(pc_feat, x_part)
            
        return final

if __name__ == '__main__':

   
    x_part = torch.randn(16, 3, 2048).cuda()
    view = torch.randn(16, 3, 224, 224).cuda()
    model = Network().cuda()
    out = model(x_part, view)
    print(out.shape) 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")
    