import pickle
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # 1. 读取两边的数据
    data_ours = load_data("/home/923714256/Data_compression_v1/SZ3_Cascade_Model/compressed_components/CurveData_Ours.pkl")
    data_neurlz = load_data("/home/923714256/Data_compression_v1/neurlz_official/temperature/CurveData_NeurLZ_1_batch.pkl")
    
    base_psnr = data_ours["base_psnr"]

    # 2. 开始绘制高颜值学术图表
    plt.figure(figsize=(10, 6), dpi=300)
    
    # 画 SZ3 基准线 (水平红虚线)
    plt.axhline(y=base_psnr, color='red', linestyle='--', linewidth=2, 
                label=f'SZ3 Base (Time: 0s, {base_psnr:.2f} dB)')

    # 画 官方 NeurLZ (蓝色，通常时间拖得很长)
    plt.plot(data_neurlz["times"], data_neurlz["psnrs"], 
             color='#1f77b4', marker='s', markersize=4, linewidth=2, 
             label=data_neurlz["name"])

    # 画 你的管线 (橙色/金色，通常时间极短，曲线像火箭一样窜上去)
    plt.plot(data_ours["times"], data_ours["psnrs"], 
             color='#ff7f0e', marker='o', markersize=5, linewidth=2.5, 
             label=data_ours["name"])

    # 3. 设置图表属性
    plt.title('Performance Trade-off: Global PSNR vs. Training Time Overhead', fontsize=14, fontweight='bold')
    plt.xlabel('Cumulative Online Training Time (Seconds)', fontsize=12)
    plt.ylabel('Compression Quality - Global PSNR (dB)', fontsize=12)
    
    # 启用网格
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # 优化布局并保存
    plt.tight_layout()
    plt.savefig('Ultimate_Pareto_Frontier_SingleField.png')
    print("🎉 终极对比图表已生成: Ultimate_Pareto_Frontier_SingleField.png")

if __name__ == "__main__":
    main()