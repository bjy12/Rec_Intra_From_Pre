import os
import sys
from turtle import forward
from numpy import append
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import _log_api_usage_once, conv1x1, conv3x3
from typing import Type, Callable, Union, List, Optional
from torch import Tensor
import pdb
from ddpm.model.modules.aggregator import localfusor, meanfusor, varfusor, adafusor

def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = norm_layer(planes * self.expansion)
        self.act = act
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.act = act
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        layers: List[int],
        feats: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dim_in = 3,
        inplanes = 64
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(dim_in, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm_layer(self.inplanes)
        self.act = act
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, act, feats[0], layers[0])
        self.layer2 = self._make_layer(block, act, feats[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, act, feats[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, act, feats[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # The function value between relu and GELU is very similar, here we directly apply "relu" initialization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last norm in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, act, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    act,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResEncoder(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.num_layers = conf.num_layers
        self.use_first_pool = conf.use_first_pool
        self.latent_size = conf.latent_size
        self.layer_num_list = conf.layer_num_list
        self.feat_num_list = conf.feat_num_list
        self.inplanes = conf.inplanes
        self.dim_in = conf.dim_in
        self.activation = conf.activation
        self.block = conf.block
        self.normalization = conf.normalization
        self.latent_volume_size = conf.latent_volume_size
        if self.activation == 'ReLU':
            act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            act = nn.GELU()
        if self.block == 'BasicBlock':
            block = BasicBlock
        elif self.block == 'Bottleneck':
            block = Bottleneck
        if self.normalization == 'Batch':
            norm = nn.BatchNorm2d
        elif self.normalization == 'Instance':
            norm = nn.InstanceNorm2d

        self.model = ResNet(block=block, act=act, layers=self.layer_num_list,
                            feats=self.feat_num_list, norm_layer=norm, inplanes=self.inplanes, dim_in=self.dim_in)
        self.aggregator_conf = conf.aggregator_conf
        self.fusion_type =  conf.fusion_type
        if self.fusion_type == 'local':
            self.aggregator = localfusor(self.aggregator_conf)
        elif self.fusion_type == 'meanmlp':
            self.aggregator = meanfusor(self.aggregator_conf)
        elif self.fusion_type == 'varmlp':
            self.aggregator = varfusor(self.aggregator_conf)
        elif self.fusion_type == 'ada':
            self.aggregator = adafusor(self.aggregator_conf)  
    
    def forward(self, x, proj_points):
        #pdb.set_trace()
        B , V , C , H , W = x.shape
        x_flat = x.view(-1, C ,H , W)
        _ , _ , N , _   = proj_points.shape
        #pdb.set_trace()
        latent_list = []
        # first layer
        x_flat = self.model.conv1(x_flat)
        x_flat = self.model.norm1(x_flat)
        x_flat = self.model.act(x_flat)
        latent_list.append(x_flat)
        if self.num_layers>=1:
            if self.use_first_pool: 
                x_flat = self.model.maxpool(x_flat)
            x_flat = self.model.layer1(x_flat)
            latent_list.append(x_flat)
        if self.num_layers>=2:
            x_flat = self.model.layer2(x_flat)
            latent_list.append(x_flat)
        if self.num_layers>=3:
            x_flat = self.model.layer3(x_flat)
            latent_list.append(x_flat)
        if self.num_layers>=4:
            x_flat = self.model.layer4(x_flat)
            latent_list.append(x_flat)
        #pdb.set_trace()
        for i in range(len(latent_list)):
            B_i , C_i , H_i , W_i = latent_list[i].shape
            latent_list[i] = latent_list[i].view(B , V , C_i , H_i , W_i)
        #pdb.set_trace()
        # feature list is multi level points wise feature by projecting back to image space        
        feature_list = self.queryfeature(latent_list , proj_points)
        #pdb.set_trace()
        multi_level_feature =  torch.cat(feature_list, dim=1) # b c n_points n_view 
        #pdb.set_trace()
        multi_level_feature =  multi_level_feature.permute(0 , 3 , 1 , 2 ) # b n_view c n_points
        #pdb.set_trace()
        agg_feature = self.aggregator(multi_level_feature) # b c n_points 
        #pdb.set_trace()
        #! reshape 是必要的? 也许 dit 可以直接融合不同token volume-wise的fusion?
        out_feature = agg_feature.reshape(B , -1 , self.latent_volume_size , self.latent_volume_size , self.latent_volume_size  )
        #pdb.set_trace()
        return out_feature
       
    def queryfeature(self, latent_list , proj_points):
        # key component of our method: feature back projection
        B , n_view , n_points , _  = proj_points.shape
        # query each level latent feature 
        #pdb.set_trace()
        feature_list = []
        #pdb.set_trace()
        for latent in latent_list:              # latent: [B, V, C, H, W]
            p_list = []
            #pdb.set_trace()
            for v in range(n_view):
                #pdb.set_trace()
                feat = latent[:, v, ...]        # [B, C, H, W]
                p = proj_points[:, v, ...]      # [B, N, 2], in [-1, 1]
                p_feats = index_2d(feat, p)     # [B, C, N]
                p_list.append(p_feats)
            #pdb.set_trace()
            p_feats_all_views = torch.stack(p_list, dim=-1)  # [B, C, N, V]
            feature_list.append(p_feats_all_views)
        #pdb.set_trace()
        return feature_list

class XRayFeatureExtractor(nn.Module):

    def __init__(self , xray_encoder_cfg):
        super().__init__()
        #pdb.set_trace()
        self.xray_encoder_cfg = xray_encoder_cfg
        self.encoder_type = xray_encoder_cfg.encoder_type
        if self.encoder_type == 'resnet':
            self.xray_encoder = ResEncoder(xray_encoder_cfg)

    def forward(self, xray_image , proj_points):
        #pdb.set_trace()
        if self.encoder_type == 'resnet':
            feature_list = self.xray_encoder(xray_image , proj_points)

                  
        return feature_list

#main 
if __name__ == '__main__':
    cfg = SimpleNamespace(
        encoder_type='resnet',
        num_layers = 3,
        use_first_pool = True,
        latent_size =128,
        layer_num_list=[3, 4, 6, 3],
        feat_num_list=[16, 32, 64, 128],
        inplanes=16,
        dim_in=1,
        activation='GELU',
        block='BasicBlock',
        normalization='Batch',
        fusion_type='ada',
        latent_volume_size=32,
        aggregator_conf = SimpleNamespace(
            latent_size=128,
            activation='GELU'
        ),
    )
    model = XRayFeatureExtractor(cfg)
    model = model.cuda()

    B, V, C, H, W = 2, 2, 1, 256, 256
    x = torch.randn(B, V, C, H, W).float().cuda()
    n_points = 32*32*32
    proj_points = torch.rand(B, V, n_points, 2) * 2 - 1
    proj_points = proj_points.float().cuda()

    feats = model(x, proj_points)

    print("Number of feature levels:", len(feats))
    for i, f in enumerate(feats):
        print(f"Level {i} shape: {f.shape}")  # [B, C, N, V]