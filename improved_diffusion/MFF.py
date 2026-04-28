import torch
import torch.nn as nn

# class MFFModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         # DiffRP 使用拼接，所以输入通道是 2 * channels
#         self.depthwise_sep_conv = nn.Sequential(
#             # 深度卷积
#             nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
#             # 逐点卷积
#             nn.Conv2d(2 * channels, channels, 1),
#             # nn.LayerNorm([channels, None, None]),
#             nn.GroupNorm(1, channels),  # 将 LayerNorm 替换为 GroupNorm(1, channels)，这实现了对整个特征图层级的归一化 
#             nn.ReLU(inplace=True)
#         )
#         self.final_conv = nn.Conv2d(channels, channels, 3, padding=1)

#     def forward(self, x_f, m_f):
#         # 1. 拼接地图特征 m_f 和 U-Net 特征 x_f
#         combined = torch.cat([x_f, m_f], dim=1)

#         # 2. 深度可分离卷积处理并加上残差连接
#         out = self.depthwise_sep_conv(combined)
#         out = out * 0.2 + x_f 
        
#         # 3. 最终卷积
#         return self.final_conv(out)

class MFFModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 1. 通道降维模块 (取代原来的 depthwise)
        # 使用 1x1 卷积是跨通道融合 x_f 和 m_f 最有效的方式
        self.channel_compress = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels), 
            nn.SiLU(inplace=True)  # 强烈建议将 ReLU 替换为 SiLU (扩散模型标配)
        )
        
        # 2. 空间特征提取模块
        # 使用标准 3x3 卷积，帮助网络理解窄墙与其周边像素的关系
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x_f, m_f):
        # 1. 绝对拼接：平等对待 噪声特征(x_f) 与 地图特征(m_f)
        combined = torch.cat([x_f, m_f], dim=1)

        # 2. 降维并动态融合：不再人为写死 0.2 的权重，让网络自己算
        out = self.channel_compress(combined)
        
        # 3. 空间提取
        out = self.spatial_conv(out)
        
        # 核心改动：绝对不要再写 "+ x_f" 了！
        # 我们要用纯净的 out 完全替代原来的特征图
        return out