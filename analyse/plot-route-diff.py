import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def calculate_diff_ratio(dir_path, save_fig_path="diff_ratio_plot.png"):
    """
    计算每个iteration的rollout和actor索引差异比例，并绘制保存图像
    
    Args:
        dir_path (str): 存放.npz文件的目标文件夹路径
        save_fig_path (str): 生成图像的保存路径
    """
    # 校验目标文件夹是否存在
    if not os.path.exists(dir_path):
        raise ValueError(f"错误：指定的文件夹不存在 -> {dir_path}")
    if not os.path.isdir(dir_path):
        raise ValueError(f"错误：指定路径不是文件夹 -> {dir_path}")

    # 收集所有.npz文件并按iteration排序
    npz_files = [f for f in os.listdir(dir_path) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"错误：文件夹 {dir_path} 中未找到.npz文件")

    # 按iteration分组（兼容文件名中iteration_后接数字的格式）
    iteration_dict = {}
    for file in npz_files:
        # 精准解析iteration数字（处理如 "rank_0_iteration_5.npz" 等格式）
        iter_start = file.find("iteration_")
        if iter_start == -1:
            print(f"警告：文件 {file} 未包含iteration_标识，已跳过")
            continue
        
        iter_part = file[iter_start + len("iteration_"):]
        # 提取iteration后的纯数字部分
        iter_num_str = ""
        for c in iter_part:
            if c.isdigit():
                iter_num_str += c
            else:
                break
        if not iter_num_str:
            print(f"警告：文件 {file} 的iteration编号无效，已跳过")
            continue
        
        try:
            iteration = int(iter_num_str)
            if iteration not in iteration_dict:
                iteration_dict[iteration] = []
            iteration_dict[iteration].append(file)
        except ValueError:
            print(f"警告：文件 {file} 的iteration编号转换失败，已跳过")
            continue

    if not iteration_dict:
        raise ValueError("错误：未识别到任何包含有效iteration_的.npz文件")
    
    sorted_iterations = sorted(iteration_dict.keys())
    diff_ratios = []  # 存储每个iteration的平均差异比例（0~1）

    # 计算每个iteration的差异比例（核心修复逻辑）
    for iter_num in sorted_iterations:
        files_in_iter = iteration_dict[iter_num]
        iter_diff_ratios = []  # 存储当前iteration下每个rank的差异比例

        for file in files_in_iter:
            file_path = os.path.join(dir_path, file)
            try:
                # 安全读取npz文件（禁用pickle避免安全风险）
                data = np.load(file_path, allow_pickle=False)
            except Exception as e:
                print(f"警告：读取文件 {file} 失败 → {e}，已跳过")
                continue

            # 校验必要键是否存在
            required_keys = ["rollout_top_indices", "actor_top_indices"]
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                print(f"警告：文件 {file} 缺少键 {missing_keys}，已跳过")
                continue

            # 提取数据并统一为数组格式（兼容标量/多维数组展平）
            rollout = np.asarray(data["rollout_top_indices"]).flatten()
            actor = np.asarray(data["actor_top_indices"]).flatten()

            # 校验数据有效性
            if len(rollout) == 0 or len(actor) == 0:
                print(f"警告：文件 {file} 索引为空，已跳过")
                continue
            if len(rollout) != len(actor):
                print(f"警告：文件 {file} 索引长度不一致（rollout:{len(rollout)}, actor:{len(actor)}），已跳过")
                continue

            # 计算单文件差异比例（核心：差异数/总元素数）
            diff_count = np.sum(rollout != actor)
            total_count = len(rollout)
            file_ratio = diff_count / total_count  # 0~1之间的比例
            iter_diff_ratios.append(file_ratio)

        # 计算当前iteration的平均差异比例
        if len(iter_diff_ratios) > 0:
            avg_ratio = np.mean(iter_diff_ratios)  # 对所有rank取平均
            diff_ratios.append(avg_ratio)
            print(f"Iteration {iter_num}: 处理{len(iter_diff_ratios)}/{len(files_in_iter)}个文件，平均差异比例 {avg_ratio:.6f}")
        else:
            diff_ratios.append(0.0)
            print(f"Iteration {iter_num}: 无有效文件，差异比例设为0")

    # 绘制并保存图像（优化可视化效果）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 兼容英文显示
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 绘制折线图（增强视觉效果）
    plt.plot(
        sorted_iterations, diff_ratios,
        marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1,
        linestyle='-', linewidth=2, color='#2E86AB', alpha=0.8
    )
    
    # 设置图表样式
    plt.xlabel("Iteration", fontsize=12, fontweight='medium')
    plt.ylabel("Difference Ratio (0~1)", fontsize=12, fontweight='medium')
    plt.title("Difference Ratio of Rollout vs Actor Indices by Iteration", fontsize=14, fontweight='bold', pad=20)
    plt.grid(alpha=0.3, linestyle='--', color='gray')
    plt.ylim(0, 0.1)  # 强制纵轴0~1（比例范围）
    plt.xticks(sorted_iterations, rotation=0)  # 显示所有iteration刻度
    plt.tight_layout()  # 自动调整布局避免标签截断

    # 保存图像（高分辨率）
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 输出汇总信息
    total_valid_files = sum(len(v) for v in iteration_dict.values())
    print("\n" + "-"*50)
    print(f"处理完成！汇总信息：")
    print(f"- 目标文件夹：{os.path.abspath(dir_path)}")
    print(f"- 图像保存路径：{os.path.abspath(save_fig_path)}")
    print(f"- 处理的Iteration数量：{len(sorted_iterations)}")
    print(f"- 扫描到的.npz文件总数：{len(npz_files)}")
    print(f"- 有效处理的文件数：{total_valid_files}")
    print(f"- 平均差异比例范围：{min(diff_ratios):.6f} ~ {max(diff_ratios):.6f}")
    print("-"*50)

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="计算NPZ文件中rollout_top_indices与actor_top_indices的差异比例并绘图",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dir", "-d", 
        required=True, 
        help="存放.npz文件的文件夹路径（必填）\n示例：--dir ./npz_data 或 -d /home/user/data"
    )
    parser.add_argument(
        "--save", "-s", 
        default="diff_ratio_plot.png", 
        help="图像保存路径（可选）\n示例：--save ./result.png 或 -s /tmp/iteration_diff.png"
    )
    
    # 解析参数并执行
    args = parser.parse_args()
    try:
        calculate_diff_ratio(dir_path=args.dir, save_fig_path=args.save)
    except Exception as e:
        print(f"\n程序执行失败：{e}")
        exit(1)