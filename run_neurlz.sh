# 1. 激活环境
source /opt/anaconda3/bin/activate grandlib

# 2. 定义数据路径
# 你可以在这里放多个文件，中间用空格隔开，AI会一个接一个处理并画对比图
TARGETS="/home/923714256/Data_compression_v1/SDRBENCH-EXASKY-NYX-512x512x512/dark_matter_density.f32 "
# 如果想测多个，可以写成：
# TARGETS="/path/to/temp.f32 /path/to/baryon_density.f32"

AUX=" /home/923714256/Data_compression_v1/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32 /home/923714256/Data_compression_v1/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32"

# 3. 运行评估脚本
python evaluate_neurlz_correct.py \
  --targets ${TARGETS} \
  --aux_data ${AUX} \
  --roi_percent 5 \
  --device cuda \
  --epochs 20 \
  --save_components ./compressed_components
