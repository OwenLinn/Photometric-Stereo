import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from src.utils import load_psm_data

def solve_poisson_dct(p, q):
    """
    使用离散余弦变换 (DCT) 快速求解泊松方程: \nabla^2 Z = \text{div}(p, q)
    这是在高分辨率下保持高度图细节的最快方法。
    """
    h, w = p.shape
    
    # 1. 计算散度 (Divergence): div = dp/dx + dq/dy
    p_x = np.zeros_like(p)
    q_y = np.zeros_like(q)
    # 使用中心差分计算梯度场散度
    p_x[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / 2
    q_y[1:-1, :] = (q[2:, :] - q[:-2, :]) / 2
    div = p_x + q_y

    # 2. 对散度进行二维 DCT 变换
    f_dct = dct(dct(div, axis=0, type=2), axis=1, type=2)

    # 3. 在频域求解
    x_coords = np.cos(np.pi * np.arange(w) / w)
    y_coords = np.cos(np.pi * np.arange(h) / h)
    ux, uy = np.meshgrid(x_coords, y_coords)
    
    # 泊松方程频域分母因子
    denom = (2 * ux + 2 * uy - 4)
    denom[0, 0] = 1.0  # 避免除以 0
    
    z_dct = f_dct / denom
    z_dct[0, 0] = 0    # 设置直流分量为 0

    # 4. 逆 DCT 变换回到空间域得到高度场 Z
    z = idct(idct(z_dct, axis=1, type=2), axis=0, type=2)
    return z

if __name__ == "__main__":
    # 路径设置
    output_dir = r"C:\Users\xingjie\Documents\VScode project\Photometric Stereo\output"
    mask_path = r"C:\Users\xingjie\Documents\VScode project\Photometric Stereo\psmImages\buddha\buddha.mask.png"
    
    # 1. 加载前两步的中间结果
    print("正在读取法线贴图和标定半径...")
    normals = np.load(os.path.join(output_dir, "buddha_normals.npy"))
    mask = cv2.imread(mask_path, 0) / 255.0
    sphere_radius = np.load(os.path.join(output_dir, "sphere_radius.npy"))[0]

    # 2. 计算表面梯度 p 和 q
    # p = dz/dx = -nx / nz, q = dz/dy = -ny / nz
    # 提高 nz 的阈值 (0.15) 可以显著平滑边缘的刺状噪声
    nz = normals[:, :, 2].copy()
    nz[nz < 0.15] = 0.15 
    p = -normals[:, :, 0] / nz
    q = -normals[:, :, 1] / nz
    
    # 仅保留 Mask 内部区域
    p[mask == 0] = 0
    q[mask == 0] = 0

    print(f"执行 DCT 积分求解高度图 (参考半径 R={sphere_radius:.2f})...")
    depth = solve_poisson_dct(p, q)
    
    # 3. 尺度对齐与归一化
    # 移除积分后的偏置，使最低点为 0
    depth_rescaled = depth - depth[mask > 0].min()
    depth_rescaled *= mask  # 再次应用 mask 确保背景干净
    
    # --- 方法 B 的应用 ---
    # 我们可以通过查看高度图的最大值与 sphere_radius 的比例来判断深度是否合理
    # 如果深度感不强，可以在这里手动乘一个系数。理论上 1.0 是物理正确的像素单位。
    final_height_map = depth_rescaled
    
    # 4. 保存可视化高度图 (8-bit PNG)
    # 将深度映射到 0-255，越亮越高
    depth_vis = (final_height_map - final_height_map.min()) / (final_height_map.max() - final_height_map.min() + 1e-6)
    depth_vis_8bit = (depth_vis * 255).astype(np.uint8)
    
    heightmap_path = os.path.join(output_dir, "buddha_heightmap.png")
    cv2.imwrite(heightmap_path, depth_vis_8bit)

    # 5. 展示结果
    plt.figure(figsize=(10, 8))
    plt.title("Reconstructed Height Map (Lighter = Higher)")
    plt.imshow(depth_vis_8bit, cmap='gray')
    plt.colorbar(label='Relative Height')
    plt.axis('off')
    plt.show()

    print(f"\n[完成] 高度图已生成: {heightmap_path}")