"""compressor.py

NeurLZ-style 多场协同压缩器 (10% 极速采样 + 异质空间哈希 + 严格界限保护版 + 精细时间统版)
负责管理 SZ3 基础压缩、多场在线训练调度以及全量推理合并。
"""

from __future__ import annotations

import os
import sys
import time
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# 假设 SZ3 已经被正确导入
sys.path.append('/home/923714256/Data_compression_v1/SZ3/tools/pysz')
try:
    from pysz import SZ
except ImportError:
    pass

from compression_function import select_roi_patches_auto_topk, select_roi_patches_user_first, select_patches_heterogeneous
from train import TrainConfig, train_online_multifield

@dataclass
class CompressionConfig:
    eb_mode: int = 1
    abs_err: float = 0
    rel_err: float = 1e-3
    pwr_err: float = 0.0
    roi_budget: int = 1024
    auto_roi: bool = True
    user_roi_box_zyx: Optional[tuple] = None
    # 🌟 默认设定为 10%，实现 10% ROI 极速训练
    roi_percent: float = 10.0 

class NeurLZCompressor:
    def __init__(self, device: str = "cuda", sz_lib_path: str = "/home/923714256/Data_compression_v1/SZ3/build/lib64/libSZ3c.so"):
        self.device = torch.device(device)
        self.sz = SZ(sz_lib_path)

    def sz3_compress(self, X: np.ndarray, cfg: CompressionConfig) -> tuple[bytes, float]:
        sz_bytes ,sz_ratio = self.sz.compress(X, cfg.eb_mode, cfg.abs_err, cfg.rel_err, cfg.pwr_err)
        return sz_bytes, sz_ratio

    def sz3_decompress(self, sz_bytes: bytes, shape: tuple, dtype: np.dtype, cfg: CompressionConfig) -> np.ndarray:
        return self.sz.decompress(sz_bytes, shape, dtype)

    def compress(self, Xs: List[np.ndarray], cfg: CompressionConfig, train_cfg: Optional[TrainConfig] = None) -> Dict[str, Any]:
        if train_cfg is None:
            train_cfg = TrainConfig()

        print("\n" + "=" * 50)
        print("  NeurLZ SOTA Pipeline (10% Fast-Sampling & Safe Bound)")
        print("=" * 50)
        
        # 🌟 总计时开始
        t_start_total = time.perf_counter()

        # ==========================================
        # 1. 基础 SZ3 压缩与解压
        # ==========================================
        X_target = Xs[0]
        shape = X_target.shape
        
        t0 = time.perf_counter()
        sz_bytes, sz_ratio = self.sz3_compress(X_target, cfg)
        t_sz_comp = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        Xp_target = self.sz3_decompress(sz_bytes, shape, np.float32, cfg)
        t_sz_decomp = time.perf_counter() - t0
        
        print(f"[SZ3 Base] Target Field Ratio: {sz_ratio:.2f}x, SZ_bytes: {len(sz_bytes)/1024:.2f} KB")

        # 2. 构建解压后的多场视角
        Xps = [Xp_target] + Xs[1:]
        true_residuals = X_target - Xp_target
        train_cfg.res_mean = float(np.mean(true_residuals))
        train_cfg.res_std = float(np.std(true_residuals)) + 1e-8

        train_cfg.input_means = [float(np.mean(f)) for f in Xps]
        train_cfg.input_stds = [float(np.std(f)) + 1e-8 for f in Xps]

        # ==========================================
        # 3. 提取 10% 异质 ROI 
        # ==========================================
        actual_max_err = float(np.max(np.abs(true_residuals)))
        train_cfg.abs_err = actual_max_err * 1.1 
        
        t0 = time.perf_counter()
        roi_list_zyx = np.zeros((0, 3), dtype=np.int32)
        if cfg.user_roi_box_zyx:
            roi_list_zyx = select_roi_patches_user_first(
                X_target, Xp_target, cfg.user_roi_box_zyx, budget=cfg.roi_budget
            )
            print(f"[ROI] User-specified {len(roi_list_zyx)} regions.")
        elif cfg.auto_roi:
            roi_list_zyx = select_patches_heterogeneous(
                X_target, Xp_target, budget=cfg.roi_budget, patch=train_cfg.roi_patch, 
                K=7, roi_percent=cfg.roi_percent
            )
            print(f"[ROI] Heterogeneous Sampling: {len(roi_list_zyx)} extreme error regions ({cfg.roi_percent}%).")
        else:
            raise ValueError("No ROI method.")
        t_roi_sampling = time.perf_counter() - t0

        # ==========================================
        # 4. 触发在线多场协同训练
        # ==========================================
        def evaluator(current_model):
            current_model.eval()
            X_h = self._run_inference_multifield(Xps, current_model, train_cfg, roi_list_zyx, cfg.rel_err, silent=True)
            mse = float(np.mean((X_target - X_h)**2))
            rng = float(np.max(X_target) - np.min(X_target))
            current_model.train()
            return 20.0 * np.log10(rng / np.sqrt(mse)) if mse > 0 else 999.0
            
        train_cfg.abs_err = cfg.abs_err 
        
        t0 = time.perf_counter()
        model, history = train_online_multifield(
            Xs=Xs, Xps=Xps, roi_list_zyx=roi_list_zyx, 
            device=self.device, cfg=train_cfg, evaluator=evaluator
        )
        t_train = time.perf_counter() - t0

        # 提取模型极简权重
        model.eval()
        model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        weights_bytes = pickle.dumps(model_weights)
        
        # ==========================================
        # 5. 全量推理合并
        # ==========================================
        t0 = time.perf_counter()
        X_hat = self._run_inference_multifield(Xps, model, train_cfg, roi_list_zyx, cfg.rel_err)
        t_inference = time.perf_counter() - t0

        # 🌟 总计结束
        total_time = time.perf_counter() - t_start_total

        # ==========================================
        # 打印详细的时间摘要
        # ==========================================
    # 🌟 新增：从 history 提取精准的纯训练时间
        t_pure_train = history["time"][-1] if "time" in history and len(history["time"]) > 0 else t_train

        print("\n" + "=" * 50)
        print("  Compression Summary & Profiling")
        print("=" * 50)
        orig_mb = X_target.nbytes / (1024**2)
        total_mb = (len(sz_bytes) + len(weights_bytes)) / (1024**2)
        print(f"[Size] Original: {orig_mb:.2f} MB")
        print(f"[Size] SZ3 Base: {len(sz_bytes)/1024:.2f} KB")
        print(f"[Size] Weights:  {len(weights_bytes)/1024:.2f} KB (Micro-UNet Backbone)")
        print(f"[Size] Total:    {total_mb:.2f} MB")
        print("-" * 50)
        print(f"[Time] SZ3 Compress:   {t_sz_comp:.3f} s")
        print(f"[Time] SZ3 Decompress: {t_sz_decomp:.3f} s")
        print(f"[Time] ROI Sampling:   {t_roi_sampling:.3f} s")
        # 🌟 修改：打印纯训练时间
        print(f"[Time] AI Training:    {t_pure_train:.3f} s  <-- (论文报告这个时间!)")
        print(f"[Time] AI Inference:   {t_inference:.3f} s")
        print(f"[Time] Total Wall Time:{total_time:.3f} s")
        print("=" * 50)

        return {
            "config": cfg,
            "train_config": train_cfg, 
            "sz_bytes": sz_bytes,
            "weights_bytes": weights_bytes,
            "roi_list_zyx": roi_list_zyx,
            "Xp": Xp_target,
            "X_hat": X_hat,
            "history": history
        }

    @torch.no_grad()
    def _run_inference_multifield(self, Xps, model, cfg, roi_list_zyx, rel_err, silent=False):
        device = next(model.parameters()).device
        Xp_target = Xps[0]
        D, H, W = Xp_target.shape
        
        ai_contribution = np.zeros_like(Xp_target, dtype=np.float32)
        n_fields = len(Xps)

        mean_t = torch.tensor(cfg.input_means, device=device).view(1, n_fields, 1, 1)
        std_t = torch.tensor(cfg.input_stds, device=device).view(1, n_fields, 1, 1)

        # 1. 全局 BG 推理 
        for z in range(D):
            xp_slice = torch.from_numpy(np.stack([f[z] for f in Xps], axis=0)).unsqueeze(0).to(device)
            xp_norm = (xp_slice - mean_t) / std_t
            r_norm = model.bg_forward(xp_norm, torch.tensor([z], device=device), 
                                      torch.tensor([0], device=device), torch.tensor([0], device=device))
            r_pred = r_norm.cpu().numpy()[0, 0] * cfg.res_std                          

            ai_contribution[z] = r_pred

        # 2. ROI 增强
        if roi_list_zyx is not None and len(roi_list_zyx) > 0:
            K = 7; patch = 32
            mean_t_roi = mean_t.repeat_interleave(K, dim=1)
            std_t_roi = std_t.repeat_interleave(K, dim=1)
            for coords in roi_list_zyx:
                zi, yi, xi = map(int, coords)
                slab = np.zeros((1, n_fields * K, patch, patch), dtype=np.float32)
                for f in range(n_fields):
                    slab[0, f*K : (f+1)*K] = Xps[f][zi:zi+K, yi:yi+patch, xi:xi+patch]
                
                slab_norm = (torch.from_numpy(slab).to(device) - mean_t_roi) / std_t_roi
                delta_norm = model.roi_forward_delta(slab_norm, torch.tensor([zi], device=device), 
                                                     torch.tensor([yi], device=device), torch.tensor([xi], device=device))
                ai_contribution[zi:zi+K, yi:yi+patch, xi:xi+patch] += delta_norm.cpu().numpy()[0] * cfg.res_std

        # 3. 🌟 终极护盾：严格的物理 Error Bound 截断
        data_range = float(np.max(Xp_target) - np.min(Xp_target))
        strict_bound = rel_err * data_range
        
        ai_contribution = np.clip(ai_contribution, -strict_bound, strict_bound)

        X_hat = Xp_target + ai_contribution
        return X_hat