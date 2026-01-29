import numpy as np
import cv2
import os
from src.utils import load_psm_data

def calibrate_lights_with_radius(images, mask):
    """
    标定光源方向，并返回铬球的像素半径用于后续深度校准 (方法 B)
    """
    # 1. 精确计算球体中心和半径
    # 使用所有 mask 为 1 的像素点坐标
    y_idx, x_idx = np.where(mask > 0)
    
    # 计算外接矩形的中心作为初值，或者使用质心
    x_c = (x_idx.min() + x_idx.max()) / 2.0
    y_c = (y_idx.min() + y_idx.max()) / 2.0
    
    # 核心：计算像素半径 R
    # 方法：取水平宽度和垂直高度的平均值
    radius_x = (x_idx.max() - x_idx.min()) / 2.0
    radius_y = (y_idx.max() - y_idx.min()) / 2.0
    radius = (radius_x + radius_y) / 2.0
    
    print(f"--- 校准信息 ---")
    print(f"球心坐标: ({x_c:.2f}, {y_c:.2f})")
    print(f"铬球像素半径 R: {radius:.2f} px")
    print(f"----------------")

    light_directions = []
    
    for i, img in enumerate(images):
        # 2. 找到最亮点的坐标 (高光)
        # 使用高斯模糊减少图像噪声对最大值定位的影响
        blur_img = cv2.GaussianBlur(img * mask, (7, 7), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blur_img)
        x_h, y_h = max_loc
        
        # 3. 计算该点在球面的单位法向量 N
        # nx, ny 是高光点相对于球心的偏移量
        nx = (x_h - x_c) / radius
        ny = (y_h - y_c) / radius
        
        # 几何约束：nx^2 + ny^2 + nz^2 = 1
        # 如果计算出的 nz < 0 (通常是高光点找错了)，强制截断为 0
        nz_sq = 1.0 - nx**2 - ny**2
        nz = np.sqrt(max(0, nz_sq))
        N = np.array([nx, ny, nz])
        
        # 4. 根据反射定律计算光源方向 L
        # V 是视线方向，假设为正交相机，即 [0, 0, 1]
        V = np.array([0, 0, 1])
        # 反射定律公式：L = 2(N·V)N - V
        L = 2 * np.dot(N, V) * N - V
        
        # 归一化光源向量确保其为单位向量
        L = L / np.linalg.norm(L)
        light_directions.append(L)
        
    return np.array(light_directions), radius

if __name__ == "__main__":
    # 配置路径
    config_path = r"C:\Users\xingjie\Documents\VScode project\Photometric Stereo\psmImages\chrome.txt"
    output_dir = r"C:\Users\xingjie\Documents\VScode project\Photometric Stereo\output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("正在加载铬球数据...")
    imgs, mask = load_psm_data(config_path)
    
    print("开始标定光源与几何尺度...")
    L_matrix, sphere_radius = calibrate_lights_with_radius(imgs, mask)
    
    # 保存光源矩阵
    np.save(os.path.join(output_dir, "light_directions.npy"), L_matrix)
    
    # 保存半径 R (方法 B 的核心参数)
    # 我们将其存为文本文件或简单的 npy 文件
    np.save(os.path.join(output_dir, "sphere_radius.npy"), np.array([sphere_radius]))
    
    print(f"\n[成功] 光源矩阵已保存")
    print(f"[成功] 几何尺度 (Radius={sphere_radius:.2f}) 已保存至 output 文件夹")