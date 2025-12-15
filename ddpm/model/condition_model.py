import os
import sys
from turtle import forward
from numpy.version import full_version
import torch
import torch.nn as nn
from ddpm.model.modules.xray_encoder import XRayFeatureExtractor


class Refine_3D_Block(nn.Module):
    def __init__(self , in_channels , out_channels , activation = 'ReLU' , norm_type = 'Batch' ):
        super(Refine_3D_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_type = norm_type

        # 1. 确定激活函数
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
            
        # 2. 确定归一化层
        if self.norm_type == 'Batch':
            NormLayer = nn.BatchNorm3d
        elif self.norm_type == 'Instance':
            NormLayer = nn.InstanceNorm3d
        else:
            raise NotImplementedError(f"Normalization type {self.norm_type} not supported")

        # 3. 构建主路径 (Conv -> Norm -> Act -> Conv -> Norm)
        # 参考 models/SRGAN.py 中的 make_res_blk 结构
        self.main_path = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            NormLayer(self.out_channels),
            self.act,
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            NormLayer(self.out_channels),
        )


    def forward(self, x):
        # 4. 执行残差连接逻辑
        residual = x
        out = self.main_path(x)
        out = out + residual
        return self.act(out)
        

class CTFeatureExtractor(nn.Module):
    def __init__(self, cfg_ct_encoder ):
        super().__init__()
        # 1. 激活函数
        self.cfg_ct_encoder = cfg_ct_encoder
        self.in_channels = cfg_ct_encoder['in_channels']
        self.out_channels = cfg_ct_encoder['out_channels']
        self.activation = cfg_ct_encoder['activation']
        self.norm_type = cfg_ct_encoder['norm_type']


        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
            
        if self.norm_type == 'Batch':
            NormLayer = nn.BatchNorm3d
        elif self.norm_type == 'Instance':
            NormLayer = nn.InstanceNorm3d
        else:
            raise NotImplementedError(f"Normalization type {self.norm_type} not supported")
        # 3. 网络结构：简单的两层卷积进行特征变换
        self.net = nn.Sequential(
            # 第一层：特征升维/变换
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            NormLayer(self.out_channels),
            self.act,
            # 第二层：进一步提炼
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            NormLayer(self.out_channels),
            self.act
        )

    def forward(self, x):
        # x shape: [B, 8, 32, 32, 32]
        return self.net(x)


class IntraXray_PreCT_Condition_Model(nn.Module):
    def __init__(self , cfg_xray_encoder , cfg_ct_encoder  ):
        super(IntraXray_PreCT_Condition_Model, self).__init__()

        #* extract xray image feature 
        self.xray_encoder = XRayFeatureExtractor(cfg_xray_encoder)
        #* extract pre operation ct feature 
        self.ct_encoder = CTFeatureExtractor(cfg_ct_encoder)
        #* 
        #self.view_embedder = nn.Embedding(num_embeddings=2, embedding_dim = view_embed_dim)

        self.refine_3d_res_blk = Refine_3D_Block(in_channels=128, out_channels=128, activation='GELU', norm_type='Batch')

        fusion_in_channels = cfg_xray_encoder.out_channels + cfg_ct_encoder.out_channels
        fusion_out_channels = 128 
        self.fusion_block = nn.Sequential(
            nn.Conv3d(fusion_in_channels, fusion_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(fusion_out_channels),
            nn.GELU(),
            # 再加一个残差块稳固融合效果（可选）
            Refine_3D_Block(fusion_out_channels, fusion_out_channels, activation='GELU')
        )        


    def forward(self, condition_dict):
        intra_ct = condition_dict['pre_ct_latent'] # b c h w d 
        projs = condition_dict['projs']
        projs_points = condition_dict['projs_points']
        #* process xray branch 
        xray_wise_feature = self.xray_encoder(projs , projs_points)
        volume_feature_from_xray = self.refine_3d_res_blk(xray_wise_feature)
        #* process pre ct branch 
        pre_ct_feature = self.ct_encoder(intra_ct)
        #* cat 
        fusion_feature = torch.cat((volume_feature_from_xray, pre_ct_feature), dim=1)
        #* 
        condition_feature = self.fusion_block(fusion_feature)

        return condition_feature