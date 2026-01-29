import numpy as np
import cv2
import os
from src.utils import load_psm_data, load_psm_data_color

def compute_brdf_components(images_color, mask, L):
    """
    SVBRDF 恢复优化版：引入鲁棒权重与彩色噪声抑制
    """
    num_imgs, h, w, _ = images_color.shape
    
    # 1. 求解几何法线 (保持之前的优化逻辑)
    images_gray = np.mean(images_color, axis=3) 
    I_gray = images_gray.reshape(num_imgs, -1)
    L_inv = np.linalg.pinv(L)
    G = np.dot(L_inv, I_gray)
    albedo_init = np.linalg.norm(G, axis=0)
    
    # 背景过滤阈值
    threshold = 0.05 
    valid_pixels = (albedo_init > threshold) & (mask.flatten() > 0)
    
    normals_flat = np.zeros((3, h * w), dtype=np.float32)
    normals_flat[2, :] = 1.0 
    normals_flat[:, valid_pixels] = G[:, valid_pixels] / albedo_init[valid_pixels]
    normals = normals_flat.T.reshape(h, w, 3) * mask[..., np.newaxis]
    
    # 2. 计算彩色反射率 (Albedo) - 引入鲁棒权重
    dot_product = np.tensordot(L, normals, axes=([1], [2])) # (12, H, W)
    albedo_color = np.zeros((h, w, 3), dtype=np.float32)
    
    for c in range(3):
        I_c = images_color[:, :, :, c]
        
        # 改进：仅使用 $N \cdot L > 0.1$ 的区域，排除阴影区的噪声放大
        # 同时排除过亮的高光点 (I < 0.9)
        weight = (dot_product > 0.1) & (I_c < 0.95) & (mask > 0)
        
        # 加权最小二乘求解
        num = np.sum(I_c * dot_product * weight, axis=0)
        den = np.sum(dot_product**2 * weight, axis=0)
        
        # 安全分母：避免极小值导致的数值爆炸
        den[den < 1e-4] = 1.0
        albedo_color[:, :, c] = (num / den) * mask

    # --- 关键步骤：彩色噪点后处理 ---
    # 使用 3x3 中值滤波去除孤立的彩色离群点
    # 将 albedo 转为 uint8 进行滤波再转回，或者直接用 cv2.medianBlur
    albedo_8bit = (np.clip(albedo_color, 0, 1) * 255).astype(np.uint8)
    albedo_filtered = cv2.medianBlur(albedo_8bit, 3) 
    albedo_color = albedo_filtered.astype(np.float32) / 255.0

    # 3. 计算镜面反射 (Specular) 与 粗糙度 (Roughness)
    diffuse_gray = np.mean(albedo_color, axis=2)
    D_theory = dot_product * diffuse_gray[np.newaxis, ...]
    residuals = images_gray - D_theory
    
    specular_map = np.max(residuals, axis=0) * mask
    specular_map = np.clip(specular_map, 0, 1)
    
    std_residuals = np.std(residuals, axis=0)
    roughness_map = (1.0 - std_residuals) * mask
    roughness_map = cv2.normalize(roughness_map, None, 0, 1, cv2.NORM_MINMAX)

    return albedo_color, normals, specular_map, roughness_map

def compute_normals_and_albedo(images, mask, L):
    """兼容旧接口的基础法线计算"""
    num_images, height, width = images.shape
    I = images.reshape(num_images, -1)
    L_inv = np.linalg.pinv(L)
    G = np.dot(L_inv, I)
    G = G.reshape(3, height, width).transpose(1, 2, 0)
    albedo = np.linalg.norm(G, axis=2)
    albedo_safe = albedo.copy(); albedo_safe[albedo == 0] = 1.0
    normals = G / albedo_safe[..., np.newaxis]
    return albedo * mask, normals * mask[..., np.newaxis]

def compute_color_albedo(images_color, mask, L, normals):
    """兼容旧接口的彩色贴图计算"""
    h, w = mask.shape
    color_albedo = np.zeros((h, w, 3))
    dot_product = np.tensordot(L, normals, axes=([1], [2]))
    for c in range(3):
        I_c = images_color[:, :, :, c]
        num = np.sum(I_c * dot_product, axis=0)
        den = np.sum(dot_product * dot_product, axis=0)
        den[den == 0] = 1.0
        color_albedo[:, :, c] = (num / den) * mask
    return np.clip(color_albedo, 0, 1)

def save_normal_map(normals, path):
    normal_bgr = ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(normal_bgr, cv2.COLOR_RGB2BGR))