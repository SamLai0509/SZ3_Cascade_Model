import time
import copy
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

# 从你自己的文件导入
from Patch_data import sample_bg_patches_multifield, sample_roi_slabs_multifield
from siren_fft_backbone_model import Cascaded_BG_ROI_Model

@dataclass
class TrainConfig:
    epochs: int = 30
    steps_per_epoch: int = 30
    bg_patch_size: int = 64
    roi_patch: int = 32
    bg_batch: int = 64
    roi_batch: int = 64
    lr: float = 1e-3
    res_mean: float = 0.0
    res_std: float = 1.0
    abs_err: float = 1.0
    input_means: list = None
    input_stds: list = None

def train_online_multifield(Xs, Xps, roi_list_zyx, device, cfg, evaluator=None):
    """
    稳定在线训练模块：完美的级联空间对齐 (Cascade Alignment)
    """
    n_fields = len(Xs)
    true_residuals = Xs[0] - Xps[0]
    
    # 🌟 用真实残差替换目标场，构造供 Patch_data 切片的列表
    Xs_for_sampling = [true_residuals] + Xs[1:]

    # 初始化级联网络
    model = Cascaded_BG_ROI_Model(
        n_fields=n_fields, 
        K=7, 
        H=512, 
        W=512
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * cfg.steps_per_epoch)
    mse_loss = nn.MSELoss()

    history = {"epoch": [], "loss": [], "psnr": [], "time": []}
    
    base_psnr = evaluator(model) if evaluator else 0.0
    history["psnr"].append((0, base_psnr)) 
    history["time"].append(0.0) 
    
    best_psnr = base_psnr                  
    best_model_weights = copy.deepcopy(model.state_dict())
    
    print(f"  [Init] Epoch   0 | Base SZ3 PSNR: {base_psnr:.2f} dB")
        
    t_start_train = time.perf_counter() 
    eval_time_total = 0.0  
    
    bg_patch_size = getattr(cfg, 'bg_patch_size', 64) 
    roi_patch_size = getattr(cfg, 'roi_patch', 32)

    for ep in range(cfg.epochs):
        model.train()
        epoch_losses = []
        
        for step in range(cfg.steps_per_epoch):
            # ==========================================
            # 1. 抓取 BG 随机切片
            # ==========================================
            bg_dict = sample_bg_patches_multifield(
                Xs=Xs_for_sampling, 
                Xps=Xps, 
                n=cfg.bg_batch,
                patch=bg_patch_size
            )
            bg_xs_t = torch.from_numpy(bg_dict["xp"]).to(device)
            bg_ys_norm = torch.from_numpy(bg_dict["x"]).to(device) / cfg.res_std
            
            bg_z_idx = torch.from_numpy(bg_dict["z"]).to(device)
            bg_y0 = torch.from_numpy(bg_dict["y0"]).to(device)
            bg_x0 = torch.from_numpy(bg_dict["x0"]).to(device)

            # ==========================================
            # 2. 抓取 ROI 切块
            # ==========================================
            roi_dict = sample_roi_slabs_multifield(
                Xs=Xs_for_sampling, 
                Xps=Xps, 
                roi_list_zyx=roi_list_zyx, 
                n=cfg.roi_batch,
                K=7,
                patch=roi_patch_size
            )
            roi_xs_t = torch.from_numpy(roi_dict["xp"]).to(device)
            roi_ys_norm = torch.from_numpy(roi_dict["x"]).to(device) / cfg.res_std
            
            roi_z0 = torch.from_numpy(roi_dict["z0"]).to(device)
            roi_y0 = torch.from_numpy(roi_dict["y0"]).to(device)
            roi_x0 = torch.from_numpy(roi_dict["x0"]).to(device)
            
            # ==========================================
            # 3. 归一化
            # ==========================================
            mean_t = torch.tensor(cfg.input_means, device=device).view(1, n_fields, 1, 1)
            std_t = torch.tensor(cfg.input_stds, device=device).view(1, n_fields, 1, 1)
            
            bg_xs_norm = (bg_xs_t - mean_t) / std_t
            
            mean_t_roi = mean_t.repeat_interleave(7, dim=1)
            std_t_roi = std_t.repeat_interleave(7, dim=1)
            roi_xs_norm = (roi_xs_t - mean_t_roi) / std_t_roi

            # ==========================================
            # 4. 前向传播与严格的级联对齐
            # ==========================================
            # (A) 训练全局 BG 网络
            bg_pred_norm = model.bg_forward(bg_xs_norm, bg_z_idx, bg_y0, bg_x0)
            loss_bg = mse_loss(bg_pred_norm, bg_ys_norm)
            
            # (B) 🌟 核心修复：精准计算 ROI 坐标处的基准预测！
            with torch.no_grad():
                bg_pred_on_roi = torch.zeros_like(roi_ys_norm) # [B, 7, 32, 32]
                for k in range(7):
                    # 极其精密：从合并的输入中抽出第 k 层所有的辅助物理场
                    indices = [f * 7 + k for f in range(n_fields)]
                    xs_k = roi_xs_norm[:, indices, :, :]
                    z_k = roi_z0 + k
                    # 让 BG 网络在正确的空间坐标上给出预测
                    bg_p = model.bg_forward(xs_k, z_k, roi_y0, roi_x0) # [B, 1, 32, 32]
                    bg_pred_on_roi[:, k:k+1, :, :] = bg_p
                    
            # (C) 训练局部 ROI 网络
            roi_delta_norm = model.roi_forward_delta(roi_xs_norm, roi_z0, roi_y0, roi_x0)
            
            # 🌟 真正的数学级联：完美对齐的基准(固定) + 增量(学习) = 目标
            roi_pred_norm = bg_pred_on_roi.detach() + roi_delta_norm 
            loss_roi = mse_loss(roi_pred_norm, roi_ys_norm)

            loss = loss_bg + loss_roi 
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
        # ==========================================
        # 5. 评估与时间剔除
        # ==========================================
        if evaluator:
            t_eval_start = time.perf_counter()
            cur_p = evaluator(model)
            t_eval_end = time.perf_counter()
            eval_time_total += (t_eval_end - t_eval_start)
            
            cum_train_time = time.perf_counter() - t_start_train - eval_time_total
            
            history["psnr"].append((ep+1, cur_p))
            history["time"].append(cum_train_time) 
            
            print(f"  Epoch {ep+1:3d} | Loss: {np.mean(epoch_losses):.6f} | PSNR: {cur_p:.2f} dB", end="")

            if cur_p > best_psnr:
                best_psnr = cur_p
                best_model_weights = copy.deepcopy(model.state_dict())
                print("  🌟 [New Best!]")
            else:
                print()

    pure_train_time = time.perf_counter() - t_start_train - eval_time_total
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\n--- 训练结束，模型已回滚至巅峰状态 (PSNR: {best_psnr:.2f} dB) ---")
        print(f"--- 💡 剔除评估开销后的纯训练耗时: {pure_train_time:.2f} 秒 ---")

    return model, history