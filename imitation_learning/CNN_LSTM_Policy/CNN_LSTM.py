import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: [B*T, C, H, W]
        attn = torch.sigmoid(self.conv1(x))  # [B*T, 1, H, W]
        return x * attn  # 广播乘法，保持维度 [B*T, C, H, W]

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, H] (H = hidden_dim)
        Q = self.query_proj(x)  # [B, T, H]
        K = self.key_proj(x)    # [B, T, H]
        V = self.value_proj(x)  # [B, T, H]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)  # [B, T, T]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, T]
        attended = torch.matmul(attn_weights, V)  # [B, T, H]
        return attended

class RGBDTemporalNetWithSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        # RGB encoder
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),   # [B*T, 3, 720, 1280] → [B*T, 32, 360, 640]
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # [B*T, 32, 180, 320]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B*T, 64, 90, 160]
            nn.ReLU(),
        )
        self.rgb_spatial_attn = SpatialAttention(64)
        self.rgb_pool = nn.AdaptiveAvgPool2d((8, 8))               # [B*T, 64, 8, 8]

        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),   # [B*T, 1, 480, 848] → [B*T, 16, 240, 424]
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # [B*T, 16, 120, 212]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B*T, 32, 60, 106]
            nn.ReLU(),
        )
        self.depth_spatial_attn = SpatialAttention(32)
        self.depth_pool = nn.AdaptiveAvgPool2d((8, 8))             # [B*T, 32, 8, 8]

        # LSTM encoder
        self.lstm = nn.LSTM(input_size=6144, hidden_size=256, batch_first=True, bidirectional=True)
        # 输入为 RGB + Depth: 64×8×8 + 32×8×8 = 6144

        # Temporal attention
        self.temporal_attn = TemporalAttention(512)

        # Action head
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # 输出动作向量 (vx, vy, wz)
        )

    def forward(self, depth, rgb):
        # depth: [B, T, 1, 480, 848]
        # rgb: [B, T, 3, 720, 1280]
        B, T = depth.shape[0], depth.shape[1]

        # flatten batch and time
        depth = depth.view(B*T, 1, 480, 848)
        rgb = rgb.view(B*T, 3, 720, 1280)

        # encode
        rgb_feat = self.rgb_encoder(rgb)           # [B*T, 64, 90, 160]
        rgb_feat = self.rgb_spatial_attn(rgb_feat) # [B*T, 64, 90, 160]
        rgb_feat = self.rgb_pool(rgb_feat)         # [B*T, 64, 8, 8]

        depth_feat = self.depth_encoder(depth)           # [B*T, 32, 60, 106]
        depth_feat = self.depth_spatial_attn(depth_feat) # [B*T, 32, 60, 106]
        depth_feat = self.depth_pool(depth_feat)         # [B*T, 32, 8, 8]

        # flatten and concat
        rgb_feat = rgb_feat.view(B, T, -1)           # [B, T, 64*8*8 = 4096]
        depth_feat = depth_feat.view(B, T, -1)       # [B, T, 32*8*8 = 2048]
        feat = torch.cat([rgb_feat, depth_feat], dim=-1)  # [B, T, 6144]

        # LSTM
        lstm_out, _ = self.lstm(feat)  # [B, T, 512]

        # Temporal attention
        attended = self.temporal_attn(lstm_out)  # [B, T, 512]

        # Predict actions per timestep
        output = self.mlp(attended)  # [B, T, 3]

        return output
