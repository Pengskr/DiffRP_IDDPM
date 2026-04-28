import torch
import torch.nn as nn

# class ResidualDenseBlock_5C(nn.Module):
#     def __init__(self, nf=64, gc=32, bias=True):
#         super(ResidualDenseBlock_5C, self).__init__()
#         # gc: intermediate channels (growth channel)
#         self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias) # 输出channel和输入channel nf 相同
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5 * 0.2 + x

# class RRDB(nn.Module):
#     def __init__(self, nf=64, gc=32):
#         super(RRDB, self).__init__()
#         self.RDB1 = ResidualDenseBlock_5C(nf, gc)
#         self.RDB2 = ResidualDenseBlock_5C(nf, gc)
#         self.RDB3 = ResidualDenseBlock_5C(nf, gc)

#     def forward(self, x):
#         out = self.RDB1(x)
#         out = self.RDB2(out)
#         out = self.RDB3(out)
#         return out * 0.2 + x

# class RRDBMapEncoder(nn.Module):
#     def __init__(self, in_nc=3, mc=64, gc=32, channel_mult=[1, 2, 4, 8]):
#         super().__init__()
#         self.conv_first = nn.Conv2d(in_nc, mc, 3, padding=1)
        
#         # 构建分层结构以匹配 U-Net 的分辨率
#         self.layers = nn.ModuleList()
#         curr_mc = mc
#         for i, mult in enumerate(channel_mult):
#             stride = 2 if i < len(channel_mult) - 1 else 1  # 只有当前层不是最后一层时，才进行 stride=2 的下采样
#             layer = nn.Sequential(
#                 RRDB(curr_mc, gc=gc),
#                 nn.Conv2d(curr_mc, mc * mult, 3, stride=stride, padding=1) # 下采样以匹配层级
#             )
#             self.layers.append(layer)
#             curr_mc = mc * mult
        
#         # 为 U-Net 中间层准备的特征提取层
#         self.middle_layer = RRDB(curr_mc, gc=gc)

#     def forward(self, m):
#         m_fea = self.conv_first(m)
#         hierarchical_features = []
#         # 提取特征 (供 MFF 注入编码器使用)
#         for layer in self.layers:
#             m_fea = layer(m_fea)
#             hierarchical_features.append(m_fea)
            
#         # 生成特征 (供中间层使用)
#         mid_fea = self.middle_layer(m_fea)
#         hierarchical_features.append(mid_fea)
        
#         return hierarchical_features


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # 注意：这里的 nf 会根据 U-Net 的层级动态变化 (如 64, 128, 256)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias) 
        
        # 改进1：使用扩散模型标配的 SiLU 替代 LeakyReLU，梯度更平滑
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        # 改进2：移除 * 0.2，保留 100% 的地图高频特征（拯救窄障碍物）
        return x5 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        
        # 同上，移除 * 0.2
        return out + x

class RRDBMapEncoder(nn.Module):
    def __init__(self, in_nc=3, mc=64, gc=32, channel_mult=[1, 2, 4, 8]):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, mc, 3, padding=1)
        
        # 改进3：将特征提取与下采样解耦，并加入通道对齐层
        self.channel_align_blocks = nn.ModuleList()
        self.rrdb_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        curr_mc = mc
        for i, mult in enumerate(channel_mult):
            # 目标通道数：匹配当前 U-Net 层级 (如 64, 128, 256)
            target_mc = mc * mult
            
            # 1. 动态对齐通道：为了能在 U-Net 下采样前完美拼接，地图编码器必须同步升维
            if curr_mc != target_mc:
                align_layer = nn.Sequential(
                    nn.Conv2d(curr_mc, target_mc, 3, padding=1),
                    nn.GroupNorm(1, target_mc),
                    nn.SiLU(inplace=True)
                )
            else:
                align_layer = nn.Identity()
            self.channel_align_blocks.append(align_layer)
            
            # 2. 在升维后的目标通道上，执行 RRDB 高清特征提取
            self.rrdb_blocks.append(RRDB(target_mc, gc=gc))
            
            # 3. 提取完特征后，再定义下采样模块
            stride = 2 if i < len(channel_mult) - 1 else 1  
            if stride == 2:
                down_layer = nn.Sequential(
                    nn.Conv2d(target_mc, target_mc, 3, stride=stride, padding=1),
                    nn.GroupNorm(1, target_mc)
                )
            else:
                down_layer = nn.Identity()
            self.downsample_blocks.append(down_layer)
            
            curr_mc = target_mc
        
        self.middle_layer = RRDB(curr_mc, gc=gc)

    def forward(self, m):
        m_fea = self.conv_first(m)
        hierarchical_features = []
        
        # 改进4：严格的 先对齐 -> 提取特征融合 -> 再下采样 流程
        for align, rrdb, downsample in zip(self.channel_align_blocks, self.rrdb_blocks, self.downsample_blocks):
            
            # A. 通道数拉升对齐 U-Net (例如从 64 变成 128)
            m_fea = align(m_fea)
            
            # B. 提取当前分辨率的特征 (例如 64x64)
            m_fea = rrdb(m_fea)
            
            # C. 核心：在下采样之前，先把最高清的特征保存下来供 U-Net 融合！
            hierarchical_features.append(m_fea)
            
            # D. 分辨率减半 (如 64x64 -> 32x32)，为下一个循环做准备
            m_fea = downsample(m_fea)
            
        # 提取最底层的中间特征供 Bottleneck 使用
        mid_fea = self.middle_layer(m_fea)
        hierarchical_features.append(mid_fea)
        
        return hierarchical_features