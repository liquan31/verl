import os
import argparse
import numpy as np

def calculate_total_diff_ratio(input_dir, target_iteration):
    """
    计算指定iteration下所有rank文件中rollout_top_indices和actor_top_indices的总差异比例
    
    Args:
        input_dir (str): 存放.npz文件的目标文件夹路径
        target_iteration (int): 要计算的目标iteration编号
    """
    # 校验目标文件夹是否存在
    if not os.path.exists(input_dir):
        raise ValueError(f"错误：指定的文件夹不存在 -> {input_dir}")
    if not os.path.isdir(input_dir):
        raise ValueError(f"错误：指定路径不是文件夹 -> {input_dir}")

    # 收集指定iteration的所有.npz文件
    npz_files = []
    for f in os.listdir(input_dir):
        if not f.endswith(".npz"):
            continue
        
        # 解析文件名中的iteration编号
        iter_start = f.find("iteration_")
        if iter_start == -1:
            continue
        
        iter_part = f[iter_start + len("iteration_"):]
        iter_num_str = ""
        for c in iter_part:
            if c.isdigit():
                iter_num_str += c
            else:
                break
        if not iter_num_str:
            continue
        
        try:
            file_iteration = int(iter_num_str)
            if file_iteration == target_iteration:
                npz_files.append(f)
        except ValueError:
            continue

    if not npz_files:
        raise FileNotFoundError(f"错误：文件夹 {input_dir} 中未找到iteration_{target_iteration}的.npz文件")
    
    print(f"找到iteration_{target_iteration}的文件数量：{len(npz_files)}")
    
    # 累计所有文件的差异数和总元素数
    total_diff_count = 0
    total_element_count = 0
    valid_file_count = 0

    # 遍历每个rank文件计算差异
    for file in npz_files:
        file_path = os.path.join(input_dir, file)
        try:
            # 安全读取npz文件（禁用pickle）
            data = np.load(file_path, allow_pickle=False)
        except Exception as e:
            print(f"读取文件 {file} 失败 → {e}，已跳过")
            continue

        # 校验必要键是否存在
        required_keys = ["rollout_top_indices", "actor_top_indices"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            print(f"文件 {file} 缺少键 {missing_keys}，已跳过")
            continue

        # 提取数据并展平为一维数组
        rollout = np.asarray(data["rollout_top_indices"]).flatten()
        actor = np.asarray(data["actor_top_indices"]).flatten()

        # 校验数据有效性
        if len(rollout) == 0 or len(actor) == 0:
            print(f"文件 {file} 索引为空，已跳过")
            continue
        if len(rollout) != len(actor):
            print(f"文件 {file} 索引长度不一致（rollout:{len(rollout)}, actor:{len(actor)}），已跳过")
            continue

        # 累计差异数和总元素数
        diff_count = np.sum(rollout != actor)
        total_count = len(rollout)
        
        total_diff_count += diff_count
        total_element_count += total_count
        valid_file_count += 1
        
        # 输出单个文件的差异信息
        file_ratio = diff_count / total_count
        print(f"   文件 {file}：差异数={diff_count}，总元素数={total_count}，差异比例={file_ratio:.6f}")

    # 计算总差异比例
    if total_element_count == 0:
        raise ValueError("错误：无有效数据用于计算总差异比例")
    
    total_diff_ratio = total_diff_count / total_element_count

    # 输出汇总信息
    print("\n" + "-"*60)
    print(f"Iteration {target_iteration} 总差异比例计算结果")
    print("-"*60)
    print(f"目标文件夹：{os.path.abspath(input_dir)}")
    print(f"扫描到的文件总数：{len(npz_files)}")
    print(f"有效处理的文件数：{valid_file_count}")
    print(f"累计总元素数：{total_element_count}")
    print(f"累计总差异数：{total_diff_count}")
    print(f"总差异比例：{total_diff_ratio:.8f} (即 {total_diff_ratio*100:.6f}%)")
    print("-"*60)

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="计算指定iteration下所有rank文件中rollout与actor索引的总差异比例",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-dir", "-d", 
        required=True, 
        help="存放.npz文件的文件夹路径（必填）\n示例：--input-dir ./npz_data 或 -d /home/user/data"
    )
    parser.add_argument(
        "--iteration", "-i", 
        type=int,
        required=True, 
        help="要计算的目标iteration编号（必填）\n示例：--iteration 5 或 -i 100"
    )
    
    # 解析参数并执行
    args = parser.parse_args()
    try:
        calculate_total_diff_ratio(input_dir=args.input_dir, target_iteration=args.iteration)
    except Exception as e:
        print(f"\n程序执行失败：{e}")
        exit(1)