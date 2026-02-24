"""train.py

NeurLZ 风格的高性能在线训练循环 (Cross-Field + High-Frequency PE 优化版)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import copy
import math

from Patch_data import sample_bg_patches_multifield, sample_roi_slabs_multifield
from siren_fft_backbone_model import Cascaded_BG_ROI_Model

# 1. 缓存 DCT 变换矩阵 (避免重复计算)
_dct_matrix_cache = {}

def get_dct_matrix(N, device):
    """生成 N x N 的 1D DCT-II 正交变换矩阵"""
    if (N, device) in _dct_matrix_cache:
        return _dct_matrix_cache[(N, device)]
    
    n = torch.arange(N, device=device).float()
    k = torch.arange(N, device=device).float().unsqueeze(1)
    C = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
    C[0, :] /= math.sqrt(2)
    C *= math.sqrt(2 / N)
    
    # 将形状调整为 (1, 1, N, N) 以便直接与 (B, C, H, W) 进行广播矩阵乘法
    _dct_matrix_cache[(N, device)] = C.unsqueeze(0).unsqueeze(0)
    return _dct_matrix_cache[(N, device)]

def dct_2d(x):
    """高效 2D DCT 变换 (基于张量矩阵乘法)"""
    H, W = x.shape[-2:]
    C_h = get_dct_matrix(H, x.device)           # (1, 1, H, H)
    C_w_t = get_dct_matrix(W, x.device).transpose(-1, -2) # (1, 1, W, W)
    
    # 2D DCT 公式: Y = C_h @ X @ C_w^T
    out = torch.matmul(C_h, x)      # 沿高度变换
    out = torch.matmul(out, C_w_t)  # 沿宽度变换
    return out

# 2. 定义纯正的 DCT 频域损失
def dct_frequency_loss(pred, target):
    pred_dct = dct_2d(pred)
    target_dct = dct_2d(target)
    # 因为 DCT 是实数，直接算 L1 绝对误差！完美保留结构特征
    return F.l1_loss(pred_dct, target_dct)

# 1. 在 TrainConfig 中加入 input 属性
@dataclass
class TrainConfig:
    epochs: int = 1
    steps_per_epoch: int = 50
    bg_batch: int = 256
    roi_batch: int = 256
    bg_patch: int = 64
    roi_patch: int = 32
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    alpha_roi: float = 0.1
    abs_err: float = 1e-3
    use_amp: bool = False
    
    # 核心新增：记录输入场和残差的全局统计量
    res_mean: float = 0.0
    res_std: float = 1.0
    input_means: list = None
    input_stds: list = None

# 2. 修改 training_step_multifield
def training_step_multifield(
    model, optimizer, Xs, Xps, roi_list_zyx, device, cfg
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = 0.0; L_bg = 0.0; L_roi = 0.0
    n_fields = len(Xs)
    
    # 构建输入特征的张量化全局参数
    mean_t = torch.tensor(cfg.input_means, device=device).view(1, n_fields, 1, 1)
    std_t = torch.tensor(cfg.input_stds, device=device).view(1, n_fields, 1, 1)

    # ==========================================
    # 1. BG 分支
    # ==========================================
    bg = sample_bg_patches_multifield(Xs, Xps, n=cfg.bg_batch, patch=cfg.bg_patch)
    if bg["xp"].shape[0] > 0:
        xp_bg = torch.from_numpy(bg["xp"]).to(device)
        x_target = torch.from_numpy(bg["x"]).to(device)
        z, y0, x0 = torch.from_numpy(bg["z"]).to(device), torch.from_numpy(bg["y0"]).to(device), torch.from_numpy(bg["x0"]).to(device)
        
        # 【输入 X' 全局标准化】
        xp_bg_norm = (xp_bg - mean_t) / std_t
        
        # 【输出残差 R 全局标准化】
        raw_res_bg = x_target - xp_bg[:, 0:1, :, :] 
        target_res_bg_norm = (raw_res_bg - cfg.res_mean) / cfg.res_std
        
        r_hat_bg_norm = model.bg_forward(xp_bg_norm, z, y0, x0)
        
        L_bg = F.mse_loss(r_hat_bg_norm, target_res_bg_norm)
        loss = loss + L_bg

    # ==========================================
    # 2. ROI 分支
    # ==========================================
    if cfg.alpha_roi > 0 and roi_list_zyx.shape[0] > 0:
        roi = sample_roi_slabs_multifield(Xs, Xps, roi_list_zyx, n=cfg.roi_batch, K=model.K, patch=cfg.roi_patch)
        if roi["xp"].shape[0] > 0:
            xp_roi = torch.from_numpy(roi["xp"]).to(device)
            x_target_roi = torch.from_numpy(roi["x"]).to(device)
            z0, y0, x0 = torch.from_numpy(roi["z0"]).to(device), torch.from_numpy(roi["y0"]).to(device), torch.from_numpy(roi["x0"]).to(device)

            # 【输入 Slab 全局标准化】需要把 mean 和 std 沿深度方向复制 K 次
            mean_t_roi = mean_t.repeat_interleave(model.K, dim=1)
            std_t_roi = std_t.repeat_interleave(model.K, dim=1)
            xp_roi_norm = (xp_roi - mean_t_roi) / std_t_roi

            with torch.no_grad():
                N_roi, _, P, P_ = xp_roi.shape
                xp_roi_2d_norm = xp_roi_norm.view(N_roi * model.K, n_fields, P, P_)
                
                z_offsets = torch.arange(model.K, device=device).unsqueeze(0).expand(N_roi, -1)
                z_all = (z0.unsqueeze(1) + z_offsets).reshape(-1)
                y0_all = y0.unsqueeze(1).expand(-1, model.K).reshape(-1)
                x0_all = x0.unsqueeze(1).expand(-1, model.K).reshape(-1)

                r_bg_base_norm = model.bg_forward(xp_roi_2d_norm, z_all, y0_all, x0_all)
                r_bg_base_norm = r_bg_base_norm.view(N_roi, model.K, P, P_) 

            r_hat_delta_norm = model.roi_forward_delta(xp_roi_norm, z0, y0, x0)

            # 【输出残差 R 全局标准化】
            raw_res_roi = x_target_roi - xp_roi[:, 0:model.K, :, :]
            target_res_roi_norm = (raw_res_roi - cfg.res_mean) / cfg.res_std
            
            L_roi = F.mse_loss(r_bg_base_norm + r_hat_delta_norm, target_res_roi_norm)
            loss = loss + cfg.alpha_roi * L_roi

    if isinstance(loss, torch.Tensor):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        return {"loss": loss.item(), "L_bg": float(L_bg), "L_roi": float(L_roi), "skipped": False}
    else:
        return {"loss": 0.0, "L_bg": 0.0, "L_roi": 0.0, "skipped": True}

# train.py (核心修复版)

def train_online_multifield(Xs, Xps, roi_list_zyx, device, cfg, evaluator=None, verbose=True):
    from siren_fft_backbone_model import Cascaded_BG_ROI_Model
    n_fields = len(Xs)
    model = Cascaded_BG_ROI_Model(n_fields=n_fields, K=7).to(device)
    
    # 调整学习率和权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    # 核心修正 2：极速预缓存 ROI (消除采样延迟)
    if verbose: print(f"--- 预缓存 {len(roi_list_zyx)} 个 ROI Slabs ---")
    roi_data = sample_roi_slabs_multifield(Xs, Xps, roi_list_zyx, n=len(roi_list_zyx), K=7, patch=32)
    roi_xp_cache = torch.from_numpy(roi_data["xp"]).to(device)
    roi_target_cache = torch.from_numpy(roi_data["x"]).to(device)
    roi_z0, roi_y0, roi_x0 = [torch.from_numpy(roi_data[k]).to(device) for k in ["z0", "y0", "x0"]]

    history = {"epoch": [], "loss": [], "psnr": []}
    mean_t = torch.tensor(cfg.input_means, device=device).view(1, n_fields, 1, 1)
    std_t = torch.tensor(cfg.input_stds, device=device).view(1, n_fields, 1, 1)

    mse_base = float(np.mean((Xs[0] - Xps[0])**2))
    rng_base = float(np.max(Xs[0]) - np.min(Xs[0]))
    base_psnr = 20.0 * np.log10(rng_base / np.sqrt(mse_base)) if mse_base > 0 else 999.0

    history["psnr"].append((0, base_psnr)) # 记录到画图数据中
    best_psnr = base_psnr                  # 将 base psnr 设为保底分数
    best_model_weights = None
    if verbose:
        print(f"  [Init] Epoch   0 | Base SZ3 PSNR: {base_psnr:.2f} dB")
    for ep in range(cfg.epochs):
        model.train()
        epoch_losses = []
        use_freq_loss =  (ep >5)
        for _ in range(cfg.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            
            # 1. BG 分支 (使用 256x256 采样)
            bg = sample_bg_patches_multifield(Xs, Xps, n=cfg.bg_batch, patch=cfg.bg_patch)
            xp_bg_norm = (torch.from_numpy(bg["xp"]).to(device) - mean_t) / std_t
            # 目标是对齐后的物理残差
            target_bg = (torch.from_numpy(bg["x"]).to(device) - torch.from_numpy(bg["xp"][:,0:1]).to(device)) / cfg.res_std
            
            pred_bg = model.bg_forward(xp_bg_norm, torch.from_numpy(bg["z"]).to(device), 
                                       torch.from_numpy(bg["y0"]).to(device), torch.from_numpy(bg["x0"]).to(device))
            # 【混合 Loss：空间 MSE + 频域 DCT L1】
            l_bg_spatial = F.mse_loss(pred_bg, target_bg)
            if use_freq_loss:
                l_bg_freq = dct_frequency_loss(pred_bg, target_bg)
                l_bg = l_bg_spatial + 0.0001 * l_bg_freq
            else:
                l_bg = l_bg_spatial

            # 2. ROI 分支 (级联训练)
            idx = torch.randperm(roi_xp_cache.size(0))[:cfg.roi_batch]
            xp_roi_norm = (roi_xp_cache[idx] - mean_t.repeat_interleave(7, dim=1)) / std_t.repeat_interleave(7, dim=1)
            
            with torch.no_grad():
                N_b = xp_roi_norm.size(0)
                # 计算 BG 的级联贡献
                r_base = model.bg_forward(xp_roi_norm.view(N_b*7, n_fields, 32, 32), 
                                          roi_z0[idx].repeat_interleave(7), 
                                          roi_y0[idx].repeat_interleave(7), 
                                          roi_x0[idx].repeat_interleave(7)).view(N_b, 7, 32, 32)
            
            delta_roi = model.roi_forward_delta(xp_roi_norm, roi_z0[idx], roi_y0[idx], roi_x0[idx])
            target_roi = (roi_target_cache[idx] - roi_xp_cache[idx, 0:7]) / cfg.res_std
            pred_roi = r_base + delta_roi
            l_roi_spatial = F.mse_loss(pred_roi, target_roi)
            if use_freq_loss:
                l_roi_freq = dct_frequency_loss(pred_roi, target_roi)
                l_roi = l_roi_spatial + 0.0001 * l_roi_freq
            else:
                l_roi = l_roi_spatial
            loss = l_bg + cfg.alpha_roi * l_roi
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        # 控制评估频率：每 5 轮评估一次
        if evaluator:
            cur_p = evaluator(model)
            history["psnr"].append((ep+1, cur_p))
            if verbose: 
                print(f"  Epoch {ep+1:3d} | Loss: {np.mean(epoch_losses):.6f} | PSNR: {cur_p:.2f} dB", end="")

            # 【核心新增 2】：如果创下新纪录，保存当前灵魂 (权重)
            if cur_p > best_psnr:
                best_psnr = cur_p
                # 因为模型极小(36KB)，deepcopy 毫无性能压力
                best_model_weights = copy.deepcopy(model.state_dict())
                if verbose: print("  🌟 [New Best!]")
            else:
                if verbose: print()

    # 【核心新增 3】：训练结束，强制回滚到最巅峰时刻！
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        if verbose: 
            print(f"\n--- 训练结束，模型已自动回滚至巅峰状态 (PSNR: {best_psnr:.2f} dB) ---")

    return model, history