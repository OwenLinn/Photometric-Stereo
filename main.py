import os
import numpy as np
import cv2
from src.step1_calibration import calibrate_lights_with_radius
from src.step2_photometric import compute_normals_and_albedo, compute_color_albedo, save_normal_map
from src.step3_integration import solve_poisson_dct
from src.utils import load_psm_data, load_psm_data_color

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_txt_files(data_dir):
    return [f for f in os.listdir(data_dir) if f.endswith('.txt')]

def main():
    # 基础路径配置
    base_path = r"C:\Users\xingjie\Documents\VScode project\Photometric Stereo"
    data_dir = os.path.join(base_path, "psmImages")
    output_dir = os.path.join(base_path, "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_obj = None
    
    while True:
        clear_screen()
        print("="*40)
        print("      光度立体 (Photometric Stereo) 工具箱")
        print("="*40)
        print(f"当前选择的对象: {selected_obj if selected_obj else '未选择'}")
        print("-"*40)
        print("1. 选择/更换数据集 (例如 buddha, cat, horse)")
        print("2. 执行 [Step 1] 光源标定 (通常选 chrome.txt)")
        print("3. 执行 [Step 2] 计算法线与反射率")
        print("4. 执行 [Step 3] 生成高度图 (Height Map)")
        print("5. 退出程序")
        print("-"*40)
        
        choice = input("请输入选项数字: ")

        if choice == '1':
            files = get_txt_files(data_dir)
            print("\n可用数据集:")
            for i, f in enumerate(files):
                print(f"  [{i}] {f}")
            idx = int(input("请选择数据集编号: "))
            selected_obj = files[idx]
            
        elif choice == '2':
            if not selected_obj:
                input("请先选择标定用的数据集（推荐 chrome.txt），按回车继续...")
                continue
            print(f"正在使用 {selected_obj} 进行光源标定...")
            imgs, mask = load_psm_data(os.path.join(data_dir, selected_obj))
            L_matrix, radius = calibrate_lights_with_radius(imgs, mask)
            np.save(os.path.join(output_dir, "light_directions.npy"), L_matrix)
            np.save(os.path.join(output_dir, "sphere_radius.npy"), np.array([radius]))
            input("\n标定成功！光源矩阵与半径已保存。按回车继续...")

        elif choice == '3':
            if not selected_obj:
                input("请先选择要处理的对象（如 buddha.txt），按回车继续...")
                continue
            
            light_path = os.path.join(output_dir, "light_directions.npy")
            if not os.path.exists(light_path):
                input("错误：找不到光源标定文件，请先执行 Step 1。")
                continue
                
            obj_name = selected_obj.replace('.txt', '')
            L = np.load(light_path)

            print(f"正在读取 {obj_name} 的彩色数据...")
            imgs_gray, mask = load_psm_data(os.path.join(data_dir, selected_obj))
            imgs_color, _ = load_psm_data_color(os.path.join(data_dir, selected_obj))

            print(f"正在计算几何法线...")
            albedo_gray, normals = compute_normals_and_albedo(imgs_gray, mask, L)
            
            print(f"正在生成彩色漫反射贴图 (Color Albedo)...")
            albedo_color = compute_color_albedo(imgs_color, mask, L, normals)

            # 保存所有结果
            cv2.imwrite(os.path.join(output_dir, f"{obj_name}_albedo_gray.png"), (albedo_gray * 255).astype(np.uint8))
            save_normal_map(normals, os.path.join(output_dir, f"{obj_name}_normal.png"))
            np.save(os.path.join(output_dir, f"{obj_name}_normals.npy"), normals)
            
            # 保存彩色贴图 (注意 OpenCV 需要 BGR 顺序保存)
            albedo_bgr = cv2.cvtColor((albedo_color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{obj_name}_albedo_color.png"), albedo_bgr)
            
            input(f"\n[成功] {obj_name} 的法线与彩色贴图已生成！按回车继续...")

        elif choice == '4':
            if not selected_obj:
                input("请先选择对象，按回车继续...")
                continue
            obj_name = selected_obj.replace('.txt', '')
            normal_path = os.path.join(output_dir, f"{obj_name}_normals.npy")
            if not os.path.exists(normal_path):
                input(f"错误：找不到 {obj_name} 的法线数据，请先执行 Step 2。")
                continue
                
            print(f"正在为 {obj_name} 恢复高度图...")
            normals = np.load(normal_path)
            # 加载 mask
            _, mask = load_psm_data(os.path.join(data_dir, selected_obj))
            
            # 梯度计算与积分
            nz = normals[:, :, 2].copy()
            nz[nz < 0.15] = 0.15 
            p, q = -normals[:, :, 0] / nz, -normals[:, :, 1] / nz
            p[mask == 0], q[mask == 0] = 0, 0
            
            depth = solve_poisson_dct(p, q)
            depth_vis = (depth - depth[mask>0].min()) * mask
            depth_norm = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-6)
            
            cv2.imwrite(os.path.join(output_dir, f"{obj_name}_heightmap.png"), (depth_norm * 255).astype(np.uint8))
            
            # 弹窗显示一下
            plt.imshow(depth_norm, cmap='gray')
            plt.title(f"{obj_name} Height Map")
            plt.show()
            input(f"\n{obj_name} 高度图生成完毕！按回车继续...")

        elif choice == '5':
            print("再见！")
            break

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()