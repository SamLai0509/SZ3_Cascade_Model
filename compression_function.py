"""compression_function.py
修正版：支持百分比动态 Budget 和用户手动 Box。
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

def _overlap_xy_ratio(a_zyx: Sequence[int], b_zyx: Sequence[int], patch: int = 32) -> float:
    _, ay, ax = a_zyx
    _, by, bx = b_zyx
    ax1, ax2 = ax, ax + patch
    ay1, ay2 = ay, ay + patch
    bx1, bx2 = bx, bx + patch
    by1, by2 = by, by + patch
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / float(patch * patch) if inter > 0 else 0.0

def nms_greedy_zyx(coords_zyx: np.ndarray, scores: np.ndarray, budget: int, 
                   patch: int = 32, K: int = 7, overlap_xy_thr: float = 0.3) -> np.ndarray:
    if coords_zyx.size == 0: return np.zeros((0, 3), dtype=np.int32)
    order = np.argsort(scores)[::-1]
    selected = []
    for idx in order:
        c = coords_zyx[idx]
        keep = True
        for s in selected:
            if abs(int(c[0]) - int(s[0])) < K and _overlap_xy_ratio(c, s, patch=patch) > overlap_xy_thr:
                keep = False
                break
        if keep:
            selected.append(c)
            if len(selected) >= budget: break
    return np.stack(selected, axis=0) if selected else np.zeros((0, 3), dtype=np.int32)

def _score_candidates_optimized(X, Xp, coords_zyx, patch=32, K=7):
    D, H, W = X.shape
    scores = np.zeros(len(coords_zyx), dtype=np.float64)
    abs_err_vol = np.abs(X - Xp)
    unique_z0 = np.unique(coords_zyx[:, 0])
    for z0 in unique_z0:
        z1 = min(D, int(z0) + K)
        E_sum = np.sum(abs_err_vol[z0:z1], axis=0)
        S = np.zeros((H + 1, W + 1), dtype=np.float64)
        S[1:, 1:] = np.cumsum(np.cumsum(E_sum, axis=0), axis=1)
        idxs = np.where(coords_zyx[:, 0] == z0)[0]
        ys, xs = coords_zyx[idxs, 1], coords_zyx[idxs, 2]
        l1_sum = S[ys+patch, xs+patch] - S[ys, xs+patch] - S[ys+patch, xs] + S[ys, xs]
        # 赋予最大误差 10 倍权重
        max_err_in_patch = np.array([np.max(abs_err_vol[z0:z1, y:y+patch, x:x+patch]) for y, x in zip(ys, xs)])
        scores[idxs] = l1_sum + 10.0 * max_err_in_patch
    return scores

def calculate_budget_by_percent(volume_shape, percent, patch_size=32, K=7):
    """关键新增：根据用户百分比计算 Patch 预算"""
    D, H, W = volume_shape
    total_voxels = D * H * W
    voxels_per_patch = K * patch_size * patch_size
    budget = int((total_voxels * (percent / 100.0)) / voxels_per_patch)
    return max(1, min(budget, 2000))

def select_roi_patches_user_first(X, Xp, user_box_zyx, budget=256, patch=32, K=7):
    """恢复缺失的函数：在用户指定的 Box 内采样。"""
    zmin, zmax, ymin, ymax, xmin, xmax = user_box_zyx
    zs = np.arange(zmin, max(zmin+1, zmax - K + 1), 2)
    ys = np.arange(ymin, max(ymin+1, ymax - patch + 1), 8)
    xs = np.arange(xmin, max(xmin+1, xmax - patch + 1), 8)
    coords = np.array([(z,y,x) for z in zs for y in ys for x in xs], dtype=np.int32)
    if coords.size == 0: return np.zeros((0, 3), dtype=np.int32)
    scores = _score_candidates_optimized(X, Xp, coords, patch, K)
    pre_select = min(len(coords), budget * 8)
    top_idx = np.argpartition(scores, -pre_select)[-pre_select:]
    return nms_greedy_zyx(coords[top_idx], scores[top_idx], budget, patch, K)

def select_roi_patches_auto_topk(X, Xp, budget=256, patch=32, K=7, roi_percent=None):
    """恢复缺失的函数：在全量空间内采样。"""
    if roi_percent:
        budget = calculate_budget_by_percent(X.shape, roi_percent, patch, K)
    D, H, W = X.shape
    zs = np.arange(0, D - K + 1, 8)
    ys = np.arange(0, H - patch + 1, 32)
    xs = np.arange(0, W - patch + 1, 32)
    coords = np.array([(z,y,x) for z in zs for y in ys for x in xs], dtype=np.int32)
    scores = _score_candidates_optimized(X, Xp, coords, patch, K)
    pre_select = min(len(coords), budget * 8)
    top_idx = np.argpartition(scores, -pre_select)[-pre_select:]
    return nms_greedy_zyx(coords[top_idx], scores[top_idx], budget, patch, K)