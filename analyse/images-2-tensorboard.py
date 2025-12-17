from torch.utils.tensorboard import SummaryWriter
import os
import cv2 as cv
import glob
import argparse
import re


# python add-png.py --input-dir /data01/ctyun/jinsd/logs/moonlight-grpo-r3-2k4k/2025-12-17_10/probs/2025-12-17_10-04-29/


# ===================== 工具函数：从文件名解析step =====================
def extract_step_from_filename(filename):
    """
    从图片文件名解析global_step（匹配step-数字格式）
    示例：scatter-step-5-121601.png → 5
    """
    step_match = re.search(r"step-(\d+)", filename)
    if step_match:
        return int(step_match.group(1))
    # 无匹配时返回默认值1
    return 1

# ===================== 工具函数：写入单张图片到TensorBoard =====================
def write_single_image(writer, img_path):
    """
    写入单张图片到TensorBoard：
    - 标签 = 图片文件名（不含后缀）
    - step = 从文件名解析的数字
    """
    # 获取图片文件名（不含路径）
    img_filename = os.path.basename(img_path)
    # 解析step
    global_step = extract_step_from_filename(img_filename)
    # 生成标签（去掉后缀，与文件名一致）
    img_tag = os.path.splitext(img_filename)[0]
    
    # 读取图片
    img = cv.imread(img_path)
    if img is None:
        print(f"跳过无法读取的图片：{img_path}")
        return
    
    # BGR转RGB（TensorBoard标准格式）
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # 写入图片
    writer.add_image(
        tag=img_tag,
        img_tensor=img_rgb,
        global_step=global_step,
        dataformats="HWC"
    )
    print(f"成功写入：{img_filename} → 标签：{img_tag}，Step：{global_step}")

# ===================== 主逻辑：解析命令行参数 + 批量处理图片 =====================
def main():
    # 1. 解析命令行参数（仅保留--input-dir，删除--tb-log-dir）
    parser = argparse.ArgumentParser(description="批量将图片写入TensorBoard（自动解析step，日志目录固定为input-dir/scatter-tensorboard）")
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="待处理图片所在文件夹（必填，如：/data01/ctyun/jinsd/logs/moonlight-grpo-r3-2k4k/2025-12-17_10/probs/2025-12-17_10-04-29/）"
    )
    args = parser.parse_args()

    # 2. 校验输入文件夹是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误：输入文件夹 {args.input_dir} 不存在！")
        return
    
    # 3. 固定TensorBoard日志目录为 input-dir/scatter-tensorboard
    tb_log_dir = os.path.join(args.input_dir, "scatter-tensorboard")
    print(f"ℹ TensorBoard日志目录：{tb_log_dir}")
    
    # 4. 创建日志目录（不存在则新建）
    os.makedirs(tb_log_dir, exist_ok=True)

    # 5. 初始化SummaryWriter
    writer = SummaryWriter(log_dir=tb_log_dir)

    try:
        # 6. 遍历文件夹下的所有图片（支持png/jpg/jpeg/bmp）
        img_extensions = (".png", ".jpg", ".jpeg", ".bmp")
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
        
        # 7. 校验是否找到图片
        if not img_paths:
            print(f"错误：在 {args.input_dir} 中未找到图片（支持格式：{img_extensions}）")
            return
        
        # 8. 批量写入图片
        print(f"\n开始处理 {len(img_paths)} 张图片...")
        for img_path in img_paths:
            write_single_image(writer, img_path)
    
    finally:
        # 强制刷盘并关闭writer，确保数据写入磁盘
        writer.flush()
        writer.close()

    # 9. 输出启动提示
    print(f"\n所有图片写入完成！")
    print(f"最终TensorBoard日志目录：{tb_log_dir}")
    print(f"\n终端执行以下命令启动TensorBoard：")
    print(f"tensorboard --logdir={tb_log_dir} --bind_all --port=8081")
    print(f"浏览器访问：http://localhost:8081（图片在「IMAGES」面板）")

if __name__ == "__main__":
    main()