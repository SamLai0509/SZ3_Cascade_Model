import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TinySkipNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=8):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.c2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(base*2, base, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c4 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.out = nn.Conv2d(base, out_ch, 1)
        
        # 核心修复：大幅增强初始化强度，让模型一开始就敢于输出大的修正值
        nn.init.normal_(self.out.weight, std=0.1) 
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x1 = F.silu(self.c1(x))
        x2 = F.silu(self.c2(x1))
        x3 = F.silu(self.c3(x2))
        x4 = torch.cat([self.up(x3), x1], dim=1)
        return self.out(F.silu(self.c4(x4)))

class Cascaded_BG_ROI_Model(nn.Module):
    def __init__(self, n_fields=1, K=7, H=512, W=512, bg_xy_bands=8, bg_z_bands=4):
        super().__init__()
        self.K = K; self.H = H; self.W = W
        self.bg_xy_bands = bg_xy_bands; self.bg_z_bands = bg_z_bands
        bg_in = n_fields + (2 + 4 * bg_xy_bands) + (1 + 2 * bg_z_bands)
        roi_in = (n_fields * K) + (2 + 4 * bg_xy_bands) + (K * (1 + 2 * bg_z_bands))
        self.bg_net = TinySkipNet(bg_in, 1, base=8)
        self.roi_net = TinySkipNet(roi_in, K, base=8)

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