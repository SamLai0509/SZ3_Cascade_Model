# Patch_data.py
import numpy as np
import torch
from typing import List, Optional, Tuple, Union

Array = np.ndarray

# ============================================================
# 核心重构：多场同步采样 (NeurLZ SOTA)
# ============================================================

def sample_bg_patches_multifield(
    Xs: List[Array],    # 原始数据场列表 [Target_X, Aux1_X, Aux2_X, ...]
    Xps: List[Array],   # 解压数据场列表 [Target_Xp, Aux1_Xp, Aux2_Xp, ...]
    n: int,
    patch: int = 64,
    seed: Optional[int] = None,
) -> dict:
    """
    同步从多个物理场采样 2D 切片。
    xp: [n, n_fields, patch, patch] -> 用于模型输入
    x_target: [n, 1, patch, patch]  -> 用于计算 Loss 的 Ground Truth
    """
    n_fields = len(Xs)
    D, H, W = Xs[0].shape
    rng = np.random.default_rng(seed)

    # 随机生成坐标 (所有场共用同一组坐标，保证物理一致性)
    z = rng.integers(0, D, size=n, dtype=np.int64)
    y0 = rng.integers(0, max(H - patch + 1, 1), size=n, dtype=np.int64)
    x0 = rng.integers(0, max(W - patch + 1, 1), size=n, dtype=np.int64)

    xp = np.zeros((n, n_fields, patch, patch), dtype=np.float32)
    x_target = np.zeros((n, 1, patch, patch), dtype=np.float32)

    for i in range(n):
        zi, yi, xi = int(z[i]), int(y0[i]), int(x0[i])
        for f in range(n_fields):
            xp[i, f] = Xps[f][zi, yi:yi+patch, xi:xi+patch]
        x_target[i, 0] = Xs[0][zi, yi:yi+patch, xi:xi+patch]

    # 清理 NaN (科学数据中常见)
    xp = np.nan_to_num(xp, nan=0.0)
    x_target = np.nan_to_num(x_target, nan=0.0)

    return {"xp": xp, "x": x_target, "z": z, "y0": y0, "x0": x0}


def sample_roi_slabs_multifield(
    Xs: List[Array],
    Xps: List[Array],
    roi_list_zyx: Array,
    n: int,
    K: int = 7,
    patch: int = 32,
    seed: Optional[int] = None,
) -> dict:
    """
    同步采样 2.5D ROI Slab。
    xp: [n, n_fields * K, patch, patch] -> 2.5D 深度方向堆叠
    x_target: [n, K, patch, patch]      -> 目标场的全量残差
    """
    n_fields = len(Xs)
    D, H, W = Xs[0].shape
    roi_list_zyx = np.asarray(roi_list_zyx)
    
    if roi_list_zyx.size == 0 or n <= 0:
        return {"xp": np.zeros((0, n_fields*K, patch, patch)), "x": np.zeros((0, K, patch, patch))}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, roi_list_zyx.shape[0], size=n)
    coords = roi_list_zyx[idx].astype(np.int64)

    z0 = np.clip(coords[:, 0], 0, max(D - K, 0))
    y0 = np.clip(coords[:, 1], 0, max(H - patch, 0))
    x0 = np.clip(coords[:, 2], 0, max(W - patch, 0))

    xp = np.zeros((n, n_fields * K, patch, patch), dtype=np.float32)
    x_target = np.zeros((n, K, patch, patch), dtype=np.float32)

    for i in range(n):
        zi, yi, xi = int(z0[i]), int(y0[i]), int(x0[i])
        # 目标场采样
        x_target[i] = Xs[0][zi:zi+K, yi:yi+patch, xi:xi+patch]
        # 多场 Slab 采样并堆叠
        for f in range(n_fields):
            slab = Xps[f][zi:zi+K, yi:yi+patch, xi:xi+patch]
            xp[i, f*K : (f+1)*K] = slab

    return {"xp": np.nan_to_num(xp), "x": np.nan_to_num(x_target), "z0": z0, "y0": y0, "x0": x0}