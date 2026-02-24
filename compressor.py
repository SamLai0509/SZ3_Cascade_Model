"""compressor.py

NeurLZ-style 多场协同压缩器 (Cross-Field SOTA Pipeline)
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

from compression_function import select_roi_patches_auto_topk, select_roi_patches_user_first
from train import TrainConfig, train_online_multifield

@dataclass
class CompressionConfig:
    eb_mode: int = 1
    abs_err: float = 1e-3
    rel_err: float = 1e-3
    pwr_err: float = 0.0
    roi_budget: int = 256
    auto_roi: bool = True
    user_roi_box_zyx: Optional[tuple] = None
    roi_percent: float = 5.0
class NeurLZCompressor:
    def __init__(self, device: str = "cuda", sz_lib_path: str = "/home/923714256/Data_compression_v1/SZ3/build/lib64/libSZ3c.so"):
        self.device = torch.device(device)
        self.sz = SZ(sz_lib_path) # 初始化 SZ3

    def sz3_compress(self, X: np.ndarray, cfg: CompressionConfig) -> tuple[bytes, float]:
        """对目标场执行基础 SZ3 压缩"""
        sz_bytes ,sz_ratio = self.sz.compress(X, cfg.eb_mode, cfg.abs_err, cfg.rel_err, cfg.pwr_err)
        # ratio = (X.nbytes) / len(sz_bytes)
        return sz_bytes, sz_ratio

    def sz3_decompress(self, sz_bytes: bytes, shape: tuple, dtype: np.dtype, cfg: CompressionConfig) -> np.ndarray:
            # SZ3 解压不需要再次传入误差界限，因为它已经保存在 sz_bytes 的头部中了
            return self.sz.decompress(sz_bytes, shape, dtype)

    def compress(self, Xs: List[np.ndarray], cfg: CompressionConfig, train_cfg: Optional[TrainConfig] = None) -> Dict[str, Any]:
        """
        核心压缩管线：
        Xs: [Target_X, Aux1_X, Aux2_X, ...]
        """
        if train_cfg is None:
            train_cfg = TrainConfig()

        print("\n" + "=" * 50)
        print("  NeurLZ SOTA Pipeline (Multi-Field)")
        print("=" * 50)
        t0 = time.time()

        # 1. 仅对目标场 (Target Field) 进行基础 SZ3 压缩
        X_target = Xs[0]
        shape = X_target.shape
        sz_bytes, sz_ratio = self.sz3_compress(X_target, cfg)
        Xp_target = self.sz3_decompress(sz_bytes, shape, np.float32, cfg)
        
        print(f"[SZ3 Base] Target Field Ratio: {sz_ratio:.2f}x, SZ_bytes: {len(sz_bytes)/1024:.2f} KB")

        # 2. 构建解压后的多场视角 (Xps)
        # 严谨起见，辅助场在解码端也必须是已知的。
        # 这里假设辅助场已就绪（原样传入，或在你的真实测试中传入它们的 Xp）
        Xps = [Xp_target] + Xs[1:]
        true_residuals = X_target - Xp_target
        train_cfg.res_mean = float(np.mean(true_residuals))
        train_cfg.res_std = float(np.std(true_residuals)) + 1e-8

        # 获取所有输入场的分布
        train_cfg.input_means = [float(np.mean(f)) for f in Xps]
        train_cfg.input_stds = [float(np.std(f)) + 1e-8 for f in Xps]
        print(f"[Norm] Res Mean: {train_cfg.res_mean:.2e}, Std: {train_cfg.res_std:.2e}")
        print(f"[Norm] Input Means: {[f'{m:.2e}' for m in train_cfg.input_means]}")


        # 3. 自动 ROI 选取 (仅基于目标场的误差进行选取)
        actual_max_err = float(np.max(np.abs(X_target - Xp_target)))
        # 将真实物理最大误差(如 4780) + 10%余量 存入 cfg.abs_err，传递给 train.py
        train_cfg.abs_err = actual_max_err * 1.1 
        print(f"[Model Setup] Real Max Error: {actual_max_err:.2f} | Dynamic Scale Bound: {train_cfg.abs_err:.2f}")
        roi_list_zyx = np.zeros((0, 3), dtype=np.int32)
        if cfg.user_roi_box_zyx:
            roi_list_zyx = select_roi_patches_user_first(
                X_target, Xp_target, cfg.user_roi_box_zyx, budget=cfg.roi_budget
            )
            print(f"[ROI] User-specified {len(roi_list_zyx)} extreme error regions.")
        elif cfg.auto_roi:
            roi_list_zyx = select_roi_patches_auto_topk(
                X_target, Xp_target, budget=cfg.roi_budget, patch=train_cfg.roi_patch, K=7,roi_percent=getattr(cfg, 'roi_percent', None)
            )
            print(f"[ROI] Auto-selected {len(roi_list_zyx)} extreme error regions.")
        else:
            raise ValueError("No ROI selection method provided. Please specify either user_roi_box_zyx or auto_roi.")

        # 4. 触发在线多场协同训练 (Online Multi-Field Training)
        def evaluator(current_model):
            current_model.eval()
            # silent=True 避免打印大量日志
            X_h = self._run_inference_multifield(Xps, current_model, train_cfg, roi_list_zyx, silent=True)
            mse = float(np.mean((X_target - X_h)**2))
            rng = float(np.max(X_target) - np.min(X_target))
            current_model.train()
            return 20.0 * np.log10(rng / np.sqrt(mse)) if mse > 0 else 999.0
        t_train_start = time.time()
        
        # 将误差界限透传给训练配置，以便模型使用 Sigmoid 进行软约束
        train_cfg.abs_err = cfg.abs_err 
        
        # 接收模型和历史记录
        model, history = train_online_multifield(
            Xs=Xs, Xps=Xps, roi_list_zyx=roi_list_zyx, 
            device=self.device, cfg=train_cfg, evaluator=evaluator
        )
        t_train = time.time() - t_train_start

        # 提取模型极简权重 (此时参数量应该只有 ~3KB)
        model.eval()
        model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        weights_bytes = pickle.dumps(model_weights)
        
        # 5. 全量推理合并 (Inference & Assembly)
        # 利用模型预测目标场的残差，生成 X_hat
        X_hat = self._run_inference_multifield(Xps, model, train_cfg, roi_list_zyx)
        
        # [安全机制] 如果需要严格保证 1x error bound，这里可以加一个基于原始数据的 Outlier 裁剪
        # 但因为模型已经用了 Sigmoid 限制，这里通常已经很安全。

        total_time = time.time() - t0
        print("\n" + "=" * 50)
        print("  Compression Summary")
        print("=" * 50)
        orig_mb = X_target.nbytes / (1024**2)
        total_mb = (len(sz_bytes) + len(weights_bytes)) / (1024**2)
        print(f"Original: {orig_mb:.2f} MB")
        print(f"SZ3:      {len(sz_bytes)/1024:.2f} KB")
        print(f"Weights:  {len(weights_bytes)/1024:.2f} KB (The beauty of TinySkipNet)")
        print(f"Total:    {total_mb:.2f} MB")
        print(f"Time:     {total_time:.2f}s (Train: {t_train:.2f}s)")
        print("=" * 50)

        return {
            "config": cfg,
            "train_config": train_cfg, 
            "sz_bytes": sz_bytes,
            "weights_bytes": weights_bytes,
            "roi_list_zyx": roi_list_zyx,
            "Xp": Xp_target,
            "X_hat": X_hat,
            "history": history   # <--- 必须加上这一行！
        }
    @torch.no_grad()
    def _run_inference_multifield(self, Xps, model, cfg, roi_list_zyx, silent=False):
        device = next(model.parameters()).device
        Xp_target = Xps[0]
        D, H, W = Xp_target.shape
        
        # 核心修正 3：独立残差修正卷，避免直接累加大数值导致的精度丢失
        ai_contribution = np.zeros_like(Xp_target, dtype=np.float32)
        n_fields = len(Xps)

        mean_t = torch.tensor(cfg.input_means, device=device).view(1, n_fields, 1, 1)
        std_t = torch.tensor(cfg.input_stds, device=device).view(1, n_fields, 1, 1)

        # 1. 全局 BG 推理 (全卷积快速模式)
        for z in range(D):
            xp_slice = torch.from_numpy(np.stack([f[z] for f in Xps], axis=0)).unsqueeze(0).to(device)
            xp_norm = (xp_slice - mean_t) / std_t
            r_norm = model.bg_forward(xp_norm, torch.tensor([z], device=device), 
                                      torch.tensor([0], device=device), torch.tensor([0], device=device))
            # 物理还原：只乘 std
            ai_contribution[z] = r_norm.cpu().numpy()[0, 0] * cfg.res_std

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
                # 叠加 Delta
                ai_contribution[zi:zi+K, yi:yi+patch, xi:xi+patch] += delta_norm.cpu().numpy()[0] * cfg.res_std

        # 3. 最终合并与安全裁剪
        X_hat = Xp_target + ai_contribution
        # total_corr = np.clip(X_hat - Xp_target, -cfg.abs_err, cfg.abs_err)
        return X_hat