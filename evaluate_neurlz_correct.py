import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from compressor import NeurLZCompressor, CompressionConfig
from train import TrainConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, nargs='+', required=True, help="数据集路径列表")
    ap.add_argument("--aux_data", type=str, nargs='*', default=[])
    ap.add_argument("--roi_box", type=str, default=None)
    ap.add_argument("--roi_percent", type=float, default=10.0) # 推荐使用 10%
    ap.add_argument("--bg_percent", type=float, default=10.0, help="BG 随机采样百分比 (隐式转换为 steps_per_epoch)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save_components", type=str, default=".") # 默认保存在当前目录
    ap.add_argument("--save_name", type=str, default="CurveData_Ours.pkl")
    args = ap.parse_args()

    all_psnr_data = {} # 存储不同数据集的曲线
    comp = NeurLZCompressor(device=args.device)

    for target_path in args.targets:
        ds_name = os.path.basename(target_path)
        print(f"\nProcessing Dataset: {ds_name}")
        
        # 1. 准备数据 (假设都是 512^3)
        X_target = np.fromfile(target_path, dtype=np.float32).reshape(512, 512, 512)
        Xs = [X_target]
        for aux in args.aux_data:
            aux_raw = np.fromfile(aux, dtype=np.float32).reshape(512, 512, 512)
            # 辅助场也可以根据需要选择是否用 log1p，这里保留你的代码
            aux_log = np.log1p(np.maximum(aux_raw, 0.0))
            Xs.append(aux_log)
        # 2. 🌟 核心：将 BG 百分比自动转换为 steps_per_epoch
        # 假设网络配置里 bg_patch_size 是 64，bg_batch 是 64
        D, H, W = X_target.shape
        bg_patch = 64
        bg_batch = 64
        
        # 计算全图能容纳多少个 64x64 的 2D 切片：512 * (512/64) * (512/64) = 32768
        total_bg_patches = D * (H // bg_patch) * (W // bg_patch)
        # 根据百分比计算目标切片数
        target_bg_patches = total_bg_patches * (args.bg_percent / 100.0)
        # 算出每轮需要跑多少次 batch
        calculated_steps = max(1, int(target_bg_patches / bg_batch))
        
        print(f"[BG Config] Requested {args.bg_percent}% BG sampling.")
        print(f"[BG Config] Auto-calculated steps_per_epoch = {calculated_steps} (Total {target_bg_patches:.0f} patches/epoch)")
        # 2. 配置
        box = tuple(map(int, args.roi_box.split(','))) if args.roi_box else None


        cfg = CompressionConfig(
            eb_mode=1,
            rel_err=1e-3,
            abs_err=0, 
            user_roi_box_zyx=box, 
            roi_percent=args.roi_percent,
            auto_roi=(box is None)
        )
        tcfg = TrainConfig(epochs=args.epochs, lr=1e-3)

        # 3. 运行
        package = comp.compress(Xs, cfg, tcfg)

        # 提取曲线数据并打包保存
        history = package["history"]
        psnrs = [p[1] for p in history["psnr"]]
        times = history["time"]
        
        save_dict = {
            "name": f"Ours ({args.roi_percent}% ROI)",
            "times": times,
            "psnrs": psnrs,
            "base_psnr": psnrs[0]
        }
        all_psnr_data[ds_name] = save_dict
        
        # 确保保存路径存在
        os.makedirs(args.save_components, exist_ok=True)
        save_path = os.path.join(args.save_components, "CurveData_Ours.pkl")    
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
    # ==========================================
    # 🌟 修复后的绘制 PSNR vs Epochs
    # ==========================================
    plt.figure(figsize=(10, 6))
    for ds_name, data_dict in all_psnr_data.items():
        # 正确提取 PSNR 数组
        vals = data_dict["psnrs"]
        # 根据 PSNR 数组长度生成 X 轴 (Epochs)
        eps = list(range(len(vals))) 
        base_psnr = data_dict["base_psnr"]
        
        # 动态绘制基准线
        plt.axhline(y=base_psnr, color='r', linestyle='--', label=f'SZ3 Base ({base_psnr:.2f} dB)')
        plt.plot(eps, vals, label=f"Data: {ds_name}", marker='o', markersize=4)

    plt.xlabel('Epochs')
    plt.ylabel('Global PSNR (dB)')
    plt.title(f'Cascaded Pipeline: PSNR Evolution (ROI: {args.roi_percent}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.save_components, 'Multi_Dataset_PSNR_Plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[Finished] 对比图已正确保存至 {plot_path}")

if __name__ == "__main__":
    main()