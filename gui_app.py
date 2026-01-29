import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np

# 导入核心逻辑
from src.utils import load_psm_data, load_psm_data_color
from src.step1_calibration import calibrate_lights_with_radius
# 确保 step2 中导入了新的 BRDF 函数
from src.step2_photometric import compute_brdf_components, save_normal_map
from src.step3_integration import solve_poisson_dct

class PhotometricStereoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Photometric Stereo SVBRDF Analyzer")
        self.root.geometry("1200x850")

        self.base_path = os.getcwd()
        self.data_dir = os.path.join(self.base_path, "psmImages")
        self.output_dir = os.path.join(self.base_path, "output")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.setup_ui()
        self.refresh_dataset_list()

    def setup_ui(self):
        # --- 左侧控制面板 ---
        control_panel = ttk.Frame(self.root, padding="15")
        control_panel.pack(side=tk.LEFT, fill=tk.Y)

        # 1. 数据集选择
        ttk.Label(control_panel, text="1. 选择数据集", font=('Helvetica', 10, 'bold')).pack(pady=(0, 5), anchor=tk.W)
        self.dataset_var = tk.StringVar()
        self.dataset_cb = ttk.Combobox(control_panel, textvariable=self.dataset_var, state="readonly")
        self.dataset_cb.pack(fill=tk.X, pady=5)
        self.dataset_cb.bind("<<ComboboxSelected>>", lambda e: self.update_display())

        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=15)

        # 2. 算法流程执行
        ttk.Label(control_panel, text="2. 执行重建流水线", font=('Helvetica', 10, 'bold')).pack(pady=5, anchor=tk.W)
        ttk.Button(control_panel, text="Step 1: 光源/尺度标定", command=self.run_step1).pack(fill=tk.X, pady=2)
        ttk.Button(control_panel, text="Step 2: 求解 SVBRDF 贴图", command=self.run_step2).pack(fill=tk.X, pady=2)
        ttk.Button(control_panel, text="Step 3: 恢复高度图", command=self.run_step3).pack(fill=tk.X, pady=2)

        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=15)

        # 3. 静态显示切换 (新增 Specular 和 Roughness)
        ttk.Label(control_panel, text="3. 材质分量预览", font=('Helvetica', 10, 'bold')).pack(pady=5, anchor=tk.W)
        self.display_var = tk.StringVar(value="Normal")
        options = ["Normal", "Albedo_Color", "Specular", "Roughness", "HeightMap"]
        for opt in options:
            ttk.Radiobutton(control_panel, text=opt, variable=self.display_var, value=opt, command=self.update_display).pack(anchor=tk.W)

        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=15)

        # 4. 实时虚拟布光
        ttk.Label(control_panel, text="4. 虚拟布光检测", font=('Helvetica', 10, 'bold')).pack(pady=5, anchor=tk.W)
        self.relight_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_panel, text="启用手电筒", variable=self.relight_mode, command=self.update_relighting).pack(pady=5, anchor=tk.W)
        
        self.az_slider = ttk.Scale(control_panel, from_=-180, to=180, orient=tk.HORIZONTAL, command=self.update_relighting)
        self.az_slider.set(0); self.az_slider.pack(fill=tk.X, pady=2)
        self.el_slider = ttk.Scale(control_panel, from_=0, to=90, orient=tk.HORIZONTAL, command=self.update_relighting)
        self.el_slider.set(45); self.el_slider.pack(fill=tk.X, pady=2)

        # --- 右侧预览面板 ---
        self.preview_panel = ttk.Frame(self.root, padding="10", relief="sunken")
        self.preview_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.img_label = ttk.Label(self.preview_panel, text="等待操作...")
        self.img_label.pack(expand=True)

    def refresh_dataset_list(self):
        if os.path.exists(self.data_dir):
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
            self.dataset_cb['values'] = sorted(files)

    def update_display(self):
        if self.relight_mode.get(): 
            self.update_relighting(); return
            
        obj_name = self.dataset_var.get().replace('.txt', '')
        if not obj_name: return

        mapping = {
            "Normal": f"{obj_name}_normal.png",
            "Albedo_Color": f"{obj_name}_albedo_color.png",
            "Specular": f"{obj_name}_specular.png",
            "Roughness": f"{obj_name}_roughness.png",
            "HeightMap": f"{obj_name}_heightmap.png"
        }
        
        img_path = os.path.join(self.output_dir, mapping[self.display_var.get()])
        if os.path.exists(img_path):
            self.show_image(img_path)
        else:
            self.img_label.config(image='', text=f"未找到: {mapping[self.display_var.get()]}")

    def show_image(self, path_or_array):
        img = Image.open(path_or_array) if isinstance(path_or_array, str) else Image.fromarray(path_or_array)
        img.thumbnail((800, 700))
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo, text="")
        self.img_label.image = photo

    def run_step1(self):
        try:
            cfg = os.path.join(self.data_dir, self.dataset_var.get())
            imgs, mask = load_psm_data(cfg)
            L, R = calibrate_lights_with_radius(imgs, mask)
            np.save(os.path.join(self.output_dir, "light_directions.npy"), L)
            np.save(os.path.join(self.output_dir, "sphere_radius.npy"), np.array([R]))
            messagebox.showinfo("成功", f"标定完成！参考半径 R={R:.2f}")
        except Exception as e: messagebox.showerror("错误", str(e))

    def run_step2(self):
        try:
            obj = self.dataset_var.get(); obj_name = obj.replace('.txt', '')
            L_path = os.path.join(self.output_dir, "light_directions.npy")
            if not os.path.exists(L_path): raise Exception("请先执行 Step 1 标定")

            L = np.load(L_path)
            imgs_c, mask = load_psm_data_color(os.path.join(self.data_dir, obj))
            
            # 调用新的 BRDF 计算函数
            print(f"正在分析 {obj_name} 的表面材质分量...")
            alb_c, normals, spec, rough = compute_brdf_components(imgs_c, mask, L)
            
            # 保存物理与可视化数据
            np.save(os.path.join(self.output_dir, f"{obj_name}_normals.npy"), normals)
            save_normal_map(normals, os.path.join(self.output_dir, f"{obj_name}_normal.png"))
            
            cv2.imwrite(os.path.join(self.output_dir, f"{obj_name}_albedo_color.png"), 
                        cv2.cvtColor((alb_c*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.output_dir, f"{obj_name}_specular.png"), (spec*255).astype(np.uint8))
            cv2.imwrite(os.path.join(self.output_dir, f"{obj_name}_roughness.png"), (rough*255).astype(np.uint8))
            
            self.update_display()
            messagebox.showinfo("成功", "SVBRDF 贴图解析完成！")
        except Exception as e: messagebox.showerror("错误", str(e))

    def run_step3(self):
        try:
            obj_name = self.dataset_var.get().replace('.txt', '')
            normal_path = os.path.join(self.output_dir, f"{obj_name}_normals.npy")
            if not os.path.exists(normal_path): raise Exception("请先执行 Step 2")

            normals = np.load(normal_path)
            _, mask = load_psm_data(os.path.join(self.data_dir, self.dataset_var.get()))
            
            nz = np.maximum(normals[:, :, 2], 0.15)
            p, q = -normals[:, :, 0]/nz, -normals[:, :, 1]/nz
            p[mask==0], q[mask==0] = 0, 0
            
            depth = solve_poisson_dct(p, q)
            depth_vis = (depth - depth[mask>0].min()) * mask
            depth_norm = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-6)
            
            cv2.imwrite(os.path.join(self.output_dir, f"{obj_name}_heightmap.png"), (depth_norm*255).astype(np.uint8))
            self.display_var.set("HeightMap"); self.update_display()
            messagebox.showinfo("成功", "高度图恢复完成")
        except Exception as e: messagebox.showerror("错误", str(e))

    def update_relighting(self, event=None):
        if not self.relight_mode.get(): self.update_display(); return
            
        obj_name = self.dataset_var.get().replace('.txt', '')
        normal_path = os.path.join(self.output_dir, f"{obj_name}_normals.npy")
        albedo_path = os.path.join(self.output_dir, f"{obj_name}_albedo_color.png")
        
        if not (os.path.exists(normal_path) and os.path.exists(albedo_path)): return

        normals = np.load(normal_path)
        albedo = cv2.imread(albedo_path)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        
        az = np.radians(float(self.az_slider.get()))
        el = np.radians(float(self.el_slider.get()))
        L_v = np.array([np.cos(el)*np.sin(az), -np.cos(el)*np.cos(az), np.sin(el)])
        
        dot = np.clip(np.einsum('ijk,k->ij', normals, L_v), 0, 1)
        # 简单渲染预览：Albedo * (N·L)
        relit = (np.clip(albedo * dot[..., np.newaxis], 0, 1) * 255).astype(np.uint8)
        self.show_image(relit)

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotometricStereoGUI(root); root.mainloop()