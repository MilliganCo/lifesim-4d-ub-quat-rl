from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def fold_uv_2x(x:torch.Tensor):
    # x: [B,C,H,W] -> [B,C,H//2,W//2,2,2]
    B,C,H,W = x.shape
    assert H%2==0 and W%2==0
    x = x.view(B,C,H//2,2,W//2,2)
    x = x.permute(0,1,2,4,3,5)  # [B,C,X,Y,U,V]
    return x

def unfold_uv_2x(x:torch.Tensor):
    # x: [B,C,X,Y,2,2] -> [B,C,H,W]
    B,C,X,Y,U,V = x.shape
    x = x.permute(0,1,2,4,3,5).contiguous().view(B,C,X*U,Y*V)
    return x

class QuaternionUVConv1x1(nn.Module):
    """Quaternion-like mixing of four UV phases via 1x1 conv on channels.
    Input: [B,C,X,Y,2,2] ; Output: same shape.
    params A,B,C,D: per-channel learnable scalars.
    y = A*x + B*roll(U) + C*roll(V) + D*roll(U,V)
    """
    def __init__(self, channels:int):
        super().__init__()
        self.A = nn.Parameter(torch.ones(channels))
        self.B = nn.Parameter(torch.zeros(channels))
        self.C = nn.Parameter(torch.zeros(channels))
        self.D = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        A,B,C,D = self.A.view(1,-1,1,1,1,1), self.B.view(1,-1,1,1,1,1), self.C.view(1,-1,1,1,1,1), self.D.view(1,-1,1,1,1,1)
        # rolls across U and V dims
        xU = torch.roll(x, shifts=1, dims=4)
        xV = torch.roll(x, shifts=1, dims=5)
        xUV = torch.roll(xU, shifts=1, dims=5)
        y = A*x + B*xU + C*xV + D*xUV
        return y

class UVQuatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim:int=512, base_channels:int=32):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.proj = nn.Conv2d(n_input_channels, base_channels, kernel_size=3, padding=1)
        self.uv1 = QuaternionUVConv1x1(base_channels)
        self.uv2 = QuaternionUVConv1x1(base_channels*2)
        self.uv3 = QuaternionUVConv1x1(base_channels*4)
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(base_channels*4, features_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        # x: [B,C,H,W]
        x = self.proj(x)
        # block 1
        uv = fold_uv_2x(x)
        uv = self.uv1(uv)
        x = unfold_uv_2x(uv)
        x = F.relu(x, inplace=True)
        # block 2
        x = self.conv2(x)
        uv = fold_uv_2x(x)
        uv = self.uv2(uv)
        x = unfold_uv_2x(uv)
        x = F.relu(x, inplace=True)
        # block 3
        x = self.conv3(x)
        uv = fold_uv_2x(x)
        uv = self.uv3(uv)
        x = unfold_uv_2x(uv)
        x = F.relu(x, inplace=True)
        return self.head(x)
