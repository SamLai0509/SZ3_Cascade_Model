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
    ap.add_argument("--roi_percent", type=float, default=5.0) # 5, 10, 15
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--save_components", type=str, default="")
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
            Xs.append(np.fromfile(aux, dtype=np.float32).reshape(512, 512, 512))

        # 2. 配置
        # 如果有 roi_box 字符串，解析成 tuple
        box = None
        if args.roi_box:
            box = tuple(map(int, args.roi_box.split(',')))

        cfg = CompressionConfig(
            abs_err=1e-3, 
            user_roi_box_zyx=box, 
            roi_percent=args.roi_percent,
            auto_roi=(box is None)
        )
        tcfg = TrainConfig(epochs=args.epochs, lr=1e-2)

        # 3. 运行
        package = comp.compress(Xs, cfg, tcfg)

        # 确保 history 键名正确
        if "history" in package:
            all_psnr_data[ds_name] = package["history"]["psnr"]

        save_path = os.path.join(args.save_components, f"NeurLZ_Comp_{ds_name}.pkl")    
        with open(save_path, 'wb') as f:
            lite_pkg = {k:v for k,v in package.items() if k not in ["X_hat", "Xp"]}
            pickle.dump(lite_pkg, f)
        
    # ==========================================
    # 绘制 PSNR vs Epochs
    # ==========================================
    plt.figure(figsize=(10, 6))
    for name, history in all_psnr_data.items():
        eps = [h[0] for h in history]
        vals = [h[1] for h in history]
        plt.plot(eps, vals, label=f"Data: {name}", marker='o', markersize=4)

    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR Evolution (ROI: {args.roi_percent}% or Manual)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_components, 'Multi_Dataset_PSNR_Plot.png'), dpi=300)
    print("\n[Finished] 对比图已保存至 Multi_Dataset_PSNR_Plot.png")

if __name__ == "__main__":
    main()