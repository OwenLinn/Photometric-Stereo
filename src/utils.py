import cv2
import numpy as np
import os

def parse_config(config_file_path):
    """
    解析官方 .txt 配置文件
    返回: (图片路径列表, mask路径, 根目录)
    """
    # 假设 main.py 在项目根目录，psmImages 也在根目录下
    # config_file_path 类似: C:/.../psmImages/buddha.txt
    # 我们的目标是获取项目根目录，以便拼接 psmImages/buddha/buddha.0.png
    base_dir = os.path.dirname(os.path.dirname(config_file_path)) 
    
    with open(config_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 解析第一行，提取图片数量 (支持 "12" 格式)
    num_images = int(lines[0].split()[-1])
    
    image_paths = lines[1:num_images + 1]
    mask_path = lines[num_images + 1]
    
    return image_paths, mask_path, base_dir

def load_psm_data(config_file_path):
    """
    标准灰度加载模式（用于计算光源和法线）
    """
    image_paths, mask_path, base_dir = parse_config(config_file_path)
    
    images = []
    for path in image_paths:
        full_path = os.path.join(base_dir, path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告：无法读取图片 {full_path}")
            continue
        # 归一化到 [0, 1] 浮点数
        images.append(img.astype(np.float32) / 255.0)
    
    mask_full_path = os.path.join(base_dir, mask_path)
    mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32) if mask is not None else None
    
    return np.array(images), mask

def load_psm_data_color(config_file_path):
    """
    彩色加载模式（用于生成彩色漫反射贴图）
    """
    image_paths, mask_path, base_dir = parse_config(config_file_path)
    
    color_images = []
    for path in image_paths:
        full_path = os.path.join(base_dir, path)
        # 读取彩色图 (BGR)
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告：无法读取图片 {full_path}")
            continue
        # 转换为 RGB 并归一化
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        color_images.append(img_rgb)
    
    mask_full_path = os.path.join(base_dir, mask_path)
    mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32) if mask is not None else None
    
    return np.array(color_images), mask