import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import matplotlib.colors as mcolors

# 自动匹配log-probs-step{数字}.npy格式的文件名，提取step
def extract_step_from_filename(filename):
    """从文件名中提取step数字，匹配log-probs-step{step}.npy格式"""
    match = re.search(r"log-probs-step(\d+)\.npy", filename)
    if match:
        return int(match.group(1))
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='批量绘制Prollout vs Ptraining的散点图')
    # 仅保留输入目录参数，移除output-dir和step参数
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='输入目录，包含所有log-probs-step*.npy文件')
    return parser.parse_args()

def process_single_file(file_path, output_dir):
    """处理单个npy文件，绘制并保存散点图"""
    # 提取文件名和step
    filename = os.path.basename(file_path)
    step = extract_step_from_filename(filename)
    if step is None:
        print(f"跳过非目标文件：{filename}（文件名不符合log-probs-step{数字}.npy格式）")
        return
    
    # 读取数据
    try:
        log_probs_dict = np.load(file_path, allow_pickle=True).item()
        rollout_logprob = log_probs_dict['rollout_log_probs']
        train_logprob = log_probs_dict['train_log_probs']
    except Exception as e:
        print(f"读取文件 {filename} 出错：{e}")
        return
    
    # 打印当前文件的基础信息
    print(f"\n===== 处理文件：{filename} (step={step}) =====")
    print("===== 原始对数概率信息 =====")
    print(f"rollout对数概率形状: {rollout_logprob.shape}")
    print(f"train对数概率形状: {train_logprob.shape}")
    print(f"rollout对数概率范围: [{np.min(rollout_logprob):.4f}, {np.max(rollout_logprob):.4f}]")
    print(f"train对数概率范围: [{np.min(train_logprob):.4f}, {np.max(train_logprob):.4f}]")
    
    # 对数概率转实际概率（数值稳定性处理）
    min_logprob = -50  # 避免极小值exp下溢
    rollout_logprob_clipped = np.clip(rollout_logprob, min_logprob, 0)
    train_logprob_clipped = np.clip(train_logprob, min_logprob, 0)
    rollout_prob = np.exp(rollout_logprob_clipped)
    train_prob = np.exp(train_logprob_clipped)
    
    # 过滤无效值（NaN/Inf）
    valid_mask = np.isfinite(rollout_prob) & np.isfinite(train_prob)
    rollout_prob = rollout_prob[valid_mask]
    train_prob = train_prob[valid_mask]
    
    if len(rollout_prob) == 0:
        print(f"文件 {filename} 过滤后无有效数据，跳过绘图")
        return
    
    # 打印转换后的概率信息
    print("\n===== 转换后实际概率信息 =====")
    print(f"有效数据点数量: {len(rollout_prob)}")
    print(f"rollout实际概率范围: [{np.min(rollout_prob):.6f}, {np.max(rollout_prob):.6f}]")
    print(f"train实际概率范围: [{np.min(train_prob):.6f}, {np.max(train_prob):.6f}]")
    diff_prob = np.abs(train_prob - rollout_prob)
    print(f"实际概率差值范围: [{np.min(diff_prob):.6f}, {np.max(diff_prob):.6f}]")
    
    # 设置颜色映射（紫色到黄色）
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'purple_to_yellow',
        [(0.2, 0.0, 0.5), (1.0, 1.0, 0.0)],
        N=256
    )
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        rollout_prob,
        train_prob,
        c=diff_prob,
        cmap=cmap,
        alpha=0.6,
        s=15,
        edgecolors='none'
    )
    
    # 绘制y=x参考线
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (Oracle)')
    
    # 坐标轴配置
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('P$_{rollout}$ ', fontsize=16)
    plt.ylabel('P$_{training}$ ', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 图例、颜色条、网格
    plt.legend(fontsize=14, loc='upper left')
    cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
    cbar.set_label('|P$_{training}$ - P$_{rollout}$| ', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 保存图片（路径：input-dir/scatter-images/scatter-step-{step}.png）
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'scatter-step-{step}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布释放内存
    print(f"图像已保存至: {save_path}")

def main():
    args = parse_args()
    
    # 1. 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误：输入目录 {args.input_dir} 不存在")
        return
    
    # 2. 创建输出子文件夹（input-dir/scatter-images）
    output_dir = os.path.join(args.input_dir, "scatter-images")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已创建/确认：{output_dir}")
    
    # 3. 遍历输入目录下所有npy文件，筛选log-probs-step*.npy格式
    npy_files = [f for f in os.listdir(args.input_dir) 
                 if f.endswith('.npy') and extract_step_from_filename(f) is not None]
    
    if not npy_files:
        print(f"输入目录 {args.input_dir} 下未找到log-probs-step*.npy格式的文件")
        return
    
    # 4. 批量处理每个文件
    print(f"\n开始处理 {len(npy_files)} 个文件...")
    for filename in npy_files:
        file_path = os.path.join(args.input_dir, filename)
        process_single_file(file_path, output_dir)
    
    print(f"\n所有文件处理完成！所有图片已保存至：{output_dir}")

if __name__ == '__main__':
    main()