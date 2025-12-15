import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class localfusor(nn.Module):
    # f_i
    def __init__(self, conf=None):
        super(localfusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc =  nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )

    def forward(self, latent):   # [nviews, C, npoints]
        feat = latent.transpose(1,2)  # [nviews, npoints, C]
        feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(feat),dim=0)
        weighted_feat = torch.sum(feat*weight,dim=0)
        output_feat = self.output_fc(weighted_feat).transpose(0,1)
        return output_feat  # [C, npoints]

class meanfusor(nn.Module):
    # f_i cat f_mean
    def __init__(self, conf=None):
        super(meanfusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc = nn.Sequential(
            nn.Linear(self.latent_size * 2 , self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
    
    def forward(self, latent):
        N = latent.shape[0]
        mean = torch.mean(latent,dim=0).repeat(N,1,1)
        feat = torch.cat([latent,mean],dim=1).transpose(1,2)
        global_feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(global_feat),dim=0)
        weighted_global_feat = torch.sum(global_feat*weight,dim=0)
        output_feat = self.output_fc(weighted_global_feat).transpose(0,1)
        return output_feat

class varfusor(nn.Module):
    # f_i cat f_var
    def __init__(self, conf=None):
        super(varfusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc = nn.Sequential(
            nn.Linear(self.latent_size * 2 , self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
    
    def forward(self, latent):
        N = latent.shape[0]
        var = torch.var(latent,dim=0).repeat(N,1,1)
        feat = torch.cat([latent,var],dim=1).transpose(1,2)
        global_feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(global_feat),dim=0)
        weighted_global_feat = torch.sum(global_feat*weight,dim=0)
        output_feat = self.output_fc(weighted_global_feat).transpose(0,1)
        return output_feat

class adafusor(nn.Module):
    # f_i cat f_mean cat f_var
    # adaptive fusion strategy proposed in our paper
    def __init__(self, conf=None):
        super(adafusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc =  nn.Sequential(
            nn.Linear(self.latent_size * 3, self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
    def forward(self, latent):
        # latent shape: [b, n_view, c, n_points]
        b, n_view, c, n_points = latent.shape
        #pdb.set_trace()
        # 1. 计算统计量 (保持维度以便广播)
        # mean/var shape: [b, 1, c, n_points] -> repeat to [b, n_view, c, n_points]
        mean = torch.mean(latent, dim=1, keepdim=True).repeat(1, n_view, 1, 1)
        var = torch.var(latent, dim=1, keepdim=True).repeat(1, n_view, 1, 1)

        # 2. 拼接特征 (关键修正)
        # 在通道维度 (dim=2) 拼接，此时通道数变为 3*c
        # feat shape: [b, n_view, 3*c, n_points]
        feat = torch.cat([latent, mean, var], dim=2)
        #pdb.set_trace()
        # 3. 维度调整以适配 Linear 层
        # Linear 需要输入特征在最后一位 (3*c)
        # [b, n_view, 3*c, n_points] -> [b, n_view, n_points, 3*c]
        feat = feat.permute(0, 1, 3, 2)

        # 4. 计算全局特征和权重
        # input_fc 输入: [..., 3*c] -> 输出: [..., c]
        # global_feat shape: [b, n_view, n_points, c]
        global_feat = self.input_fc(feat)

        # 计算权重，并在视角维度 (dim=1) 进行 Softmax
        # weight shape: [b, n_view, n_points, 1] (假设 weight_fc 输出 1 通道)
        weight = F.softmax(self.weight_fc(global_feat), dim=1)

        # 5. 加权融合
        # 对 n_view 维度求和，消除该维度
        # weighted_global_feat shape: [b, n_points, c]
        weighted_global_feat = torch.sum(global_feat * weight, dim=1)

        # 6. 输出映射与最终维度调整
        # output_fc 输入: [b, n_points, c] -> 输出: [b, n_points, c]
        output_feat = self.output_fc(weighted_global_feat)

        # 调整回 [b, c, n_points] 以匹配常规 CNN/PointNet 格式
        output_feat = output_feat.transpose(1, 2)
        #pdb.set_trace()
        
        return output_feat