import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import BasicUNet

def fourier_xy_abs(y0, x0, ph, pw, H, W, num_bands):
    device = y0.device
    u = torch.arange(ph, device=device).float()
    v = torch.arange(pw, device=device).float()
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    yy = y0[:, None, None].float() + uu[None, :, :]
    xx = x0[:, None, None].float() + vv[None, :, :]
    y = 2.0 * (yy / max(H - 1, 1)) - 1.0
    x = 2.0 * (xx / max(W - 1, 1)) - 1.0
    feats = [x[:, None], y[:, None]]
    freqs = (2.0 ** torch.arange(num_bands, device=device)) * math.pi
    for f in freqs:
        feats += [torch.sin(x[:, None] * f), torch.cos(x[:, None] * f),
                  torch.sin(y[:, None] * f), torch.cos(y[:, None] * f)]
    return torch.cat(feats, dim=1)

def fourier_z_1d(z, num_bands):
    device = z.device
    feats = [z[:, None]]
    freqs = (2.0 ** torch.arange(num_bands, device=device)) * math.pi
    zfb = z[:, None] * freqs[None, :]
    feats += [torch.sin(zfb), torch.cos(zfb)]
    return torch.cat(feats, dim=1)

# class Sine(nn.Module):
#     def __init__(self, w0=30.0):
#         super().__init__()
#         self.w0 = w0

#     def forward(self, x):
#         # SIREN 核心公式: sin(w0 * x)
#         return torch.sin(self.w0 * x)

# class SirenConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
#                  is_first=False, w0=30.0):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
#                               stride=stride, padding=padding)
#         self.is_first = is_first
#         self.w0 = w0
#         self.activation = Sine(w0)
        
#         self.init_weights()

#     def init_weights(self):
#         with torch.no_grad():
#             # 对于 Conv2d，fan_in = in_channels * kernel_size * kernel_size
#             fan_in = self.conv.weight.size(1) * self.conv.weight.size(2) * self.conv.weight.size(3)
            
#             if self.is_first:
#                 # 论文要求第一层初始化
#                 bound = 1.0 / fan_in
#             else:
#                 # 论文要求隐藏层初始化: sqrt(6 / fan_in) / w0
#                 bound = math.sqrt(6.0 / fan_in) / self.w0
                
#             self.conv.weight.uniform_(-bound, bound)
#             if self.conv.bias is not None:
#                 nn.init.zeros_(self.conv.bias)

#     def forward(self, x):
#         return self.activation(self.conv(x))

# class TinySirenSkipNet(nn.Module):
#     # w0=30.0 是论文推荐值，针对坐标输入效果最好
#     def __init__(self, in_ch, out_ch, base=8, w0=30.0):
#         super().__init__()
        
#         # 只有网络的第一层 is_first=True
#         self.c1 = SirenConv2d(in_ch, base, 3, padding=1, is_first=True, w0=w0)
#         self.c2 = SirenConv2d(base, base*2, 3, stride=2, padding=1, w0=w0)
#         self.c3 = SirenConv2d(base*2, base, 3, padding=1, w0=w0)
        
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.c4 = SirenConv2d(base * 2, base, 3, padding=1, w0=w0)
        
#         # 最后一层【绝对不能】加 Sine，直接输出物理修正值
#         self.out = nn.Conv2d(base, out_ch, 1)
        
#         # 保持我们之前设计的“破冰”强力初始化
#         nn.init.normal_(self.out.weight, std=0.1) 
#         nn.init.zeros_(self.out.bias)

#     def forward(self, x):
#         x1 = self.c1(x)
#         x2 = self.c2(x1)
#         x3 = self.c3(x2)
#         x4 = torch.cat([self.up(x3), x1], dim=1)
#         return self.out(self.c4(x4))
class Micro_ConvNeXt_Block(nn.Module):
    """
    极简版 ConvNeXt Block，专为科学数据高频激波拟合设计。
    包含: 7x7 大核深度可分离卷积 + 4倍倒置瓶颈 + GELU
    【关键约束】：严格去除了 LayerNorm，保持全局平移和尺度不变性 (Scale-Invariant)，誓死保卫背景精度！
    """
    def __init__(self, dim):
        super().__init__()
        # 1. 7x7 Depthwise Conv: 用极小的参数量换取巨大的空间感受野
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. 1x1 Pointwise Conv: 特征升维 4 倍 (Inverted Bottleneck)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        
        # 3. 官方指定的高级激活函数
        self.act = nn.GELU()
        
        # 4. 1x1 Pointwise Conv: 特征降维回原尺度
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input_x + x  # 纯粹的残差连接


class Micro_ConvNeXt_UNet_Backbone(nn.Module):
    """
    结合了 U-Net 的空间跳跃连接 (Skip) 与 ConvNeXt 的超强特征提取。
    """
    def __init__(self, in_channels, out_channels, hidden=16):
        super().__init__()
        # 初始投影
        self.inc = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        
        # Encoder (大核特征提取)
        self.enc_block1 = Micro_ConvNeXt_Block(hidden)
        
        self.down1 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.enc_block2 = Micro_ConvNeXt_Block(hidden)
        
        self.down2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.bottleneck = Micro_ConvNeXt_Block(hidden)
        
        # Decoder 与跳跃连接 (Skip Connections)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv1 = nn.Conv2d(hidden * 2, hidden, kernel_size=1) # 用 1x1 极速融合跳跃特征
        self.dec_block1 = Micro_ConvNeXt_Block(hidden)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv2 = nn.Conv2d(hidden * 2, hidden, kernel_size=1) 
        self.dec_block2 = Micro_ConvNeXt_Block(hidden)
        
        # 物理残差输出头
        self.outc = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)

        # 🌟 严苛的物理界限保护：零初始化
        nn.init.zeros_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1_c = self.enc_block1(x1)
        
        x2 = self.down1(x1_c)
        x2_c = self.enc_block2(x2)
        
        x3 = self.down2(x2_c)
        x3_c = self.bottleneck(x3)
        
        # Decoder with Skip Connections
        u1 = self.up1(x3_c)
        u1 = torch.cat([u1, x2_c], dim=1) # 跳跃拼接 [B, hidden*2, H/2, W/2]
        u1 = self.up_conv1(u1)            # 降维回 hidden
        u1_c = self.dec_block1(u1)
        
        u2 = self.up2(u1_c)
        u2 = torch.cat([u2, x1_c], dim=1)
        u2 = self.up_conv2(u2)
        u2_c = self.dec_block2(u2)
        
        return self.outc(u2_c)



class Micro_UNet_Backbone(nn.Module):
    """
    专为你的 Patch 训练设计的极简 U-Net。
    完美复刻 NeurLZ 核心优势：
    1. 极低参数量 (~5000) 免疫量化噪声过拟合
    2. 无任何归一化层 (Scale-Invariant) 避免全图推理崩溃
    3. 跳跃连接 (Skip) 平滑高频边界
    """
    def __init__(self, in_channels, out_channels, hidden=8):
        super().__init__()
        # 初始特征提取
        self.inc = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        
        # 下采样层 (扩大感受野)
        self.down1 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        
        # 上采样与跳跃连接
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv1 = nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv2 = nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1)
        
        # 物理残差输出头
        self.outc = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
        
        # 官方 NeurLZ 御用激活函数
        self.act = nn.GELU()

        # 🌟 严苛的物理界限保护：零初始化
        nn.init.zeros_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        # Encoder
        x1 = self.act(self.inc(x))
        x2 = self.act(self.down1(x1))
        x3 = self.act(self.down2(x2))
        
        # Decoder with Skip Connections
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1) # 跳跃特征拼接
        u1 = self.act(self.up_conv1(u1))
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1) # 跳跃特征拼接
        u2 = self.act(self.up_conv2(u2))
        
        return self.outc(u2)


        
class Cascaded_BG_ROI_Model(nn.Module):
    def __init__(self, n_fields=1, K=7, H=512, W=512, bg_xy_bands=8, bg_z_bands=4):
        super().__init__()
        self.K = K; self.H = H; self.W = W
        self.bg_xy_bands = bg_xy_bands; self.bg_z_bands = bg_z_bands
        
        # 🌟 恢复高频坐标：打破 CNN 的平移不变性，锚定 SZ3 的误差网格！
        bg_in = n_fields + (2 + 4 * bg_xy_bands) + (1 + 2 * bg_z_bands)
        roi_in = (n_fields * K) + (2 + 4 * bg_xy_bands) + (K * (1 + 2 * bg_z_bands))
        
        # self.bg_net = BasicUNet(
        #     spatial_dims=2,
        #     features=(4, 4, 4, 4, 4, 4),
        #     act='gelu',
        #     in_channels=bg_in,
        #     out_channels=1
        # )
        # self.roi_net = BasicUNet(
        #     spatial_dims=2,
        #     features=(4, 4, 4, 4, 4, 4),
        #     act='gelu',
        #     in_channels=roi_in,
        #     out_channels=K,
        # )
        self.bg_net = Micro_UNet_Backbone(in_channels=bg_in, out_channels=1, hidden=16)
        self.roi_net = Micro_UNet_Backbone(in_channels=roi_in, out_channels=K, hidden=16)

    def bg_forward(self, xp_all_fields, z_idx, y0, x0):
        N, _, ph, pw = xp_all_fields.shape
        xy = fourier_xy_abs(y0, x0, ph, pw, self.H, self.W, self.bg_xy_bands)
        z = 2.0 * (z_idx.float() / (self.H - 1)) - 1.0
        zf = fourier_z_1d(z, self.bg_z_bands).view(N, -1, 1, 1).expand(-1, -1, ph, pw)
        return self.bg_net(torch.cat([xp_all_fields, xy, zf], dim=1))

    def roi_forward_delta(self, slab_all_fields, z_idx_base, y0, x0):
        N, _, ph, pw = slab_all_fields.shape
        xy = fourier_xy_abs(y0, x0, ph, pw, self.H, self.W, self.bg_xy_bands)
        z_offsets = torch.arange(self.K, device=slab_all_fields.device).float()
        z_coords = 2.0 * ((z_idx_base[:, None] + z_offsets[None, :]) / (self.H - 1)) - 1.0
        zf = fourier_z_1d(z_coords.view(-1), self.bg_z_bands)
        zf = zf.view(N, self.K * zf.shape[1], 1, 1).expand(-1, -1, ph, pw)
        return self.roi_net(torch.cat([slab_all_fields, xy, zf], dim=1))