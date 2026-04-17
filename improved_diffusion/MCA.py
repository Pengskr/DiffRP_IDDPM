import torch
import torch.nn as nn
import torch.nn.functional as F

class MACModule(nn.Module):
    """
    Map-Conditioned Attention (MAC) Module
    集成 Spatial Attention 和 Channel Attention，基于 PP-LiteSeg UAFM 架构。
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # --- Spatial Attention Branch ---
        # 输入 4 通道: [mean(x_f), max(x_f), mean(m_f), max(m_f)]
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # --- Channel Attention Branch ---
        # 输入 4*C 通道: [avg(x_f), max(x_f), avg(m_f), max(m_f)] -> 压缩为 1x1
        # 这里参考 UAFM ChAtten 逻辑，使用 1x1 卷积生成通道权重
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_f, m_f):
        """
        x_f: U-Net 中间层特征 [B, C, H, W]
        m_f: RRDB middle_layer 输出 [B, C, H, W]
        """
        # 1. 计算 Spatial Attention 权重 (alpha)
        # 对两个输入分别做 MEAN 和 MAX
        s_x_avg = torch.mean(x_f, dim=1, keepdim=True)
        s_x_max = torch.max(x_f, dim=1, keepdim=True)[0]
        s_m_avg = torch.mean(m_f, dim=1, keepdim=True)
        s_m_max = torch.max(m_f, dim=1, keepdim=True)[0]
        
        # 拼接并计算权重
        alpha = self.spatial_conv(torch.cat([s_x_avg, s_x_max, s_m_avg, s_m_max], dim=1))
        
        # 2. 计算 Channel Attention 权重 (beta)
        # 对两个输入分别做全局 Avg 和 Max 池化
        c_x_avg = F.adaptive_avg_pool2d(x_f, 1)
        c_x_max = F.adaptive_max_pool2d(x_f, 1)
        c_m_avg = F.adaptive_avg_pool2d(m_f, 1)
        c_m_max = F.adaptive_max_pool2d(m_f, 1)
        
        # 拼接并计算权重 (B, 4C, 1, 1)
        beta = self.channel_conv(torch.cat([c_x_avg, c_x_max, c_m_avg, c_m_max], dim=1))
        
        # 3. 特征融合 (参考 UAFM: F = alpha * high + (1-alpha) * low)
        # 这里我们结合空间和通道注意力
        # 先应用空间融合
        feat_spatial = x_f * alpha + m_f * (1.0 - alpha)
        
        # 再应用通道重校准
        out = feat_spatial * beta + m_f * (1.0 - beta)
        
        return out