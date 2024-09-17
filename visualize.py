import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_cam_on_image(img, cam, output_path=None):
    # 归一化 CAM
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # 将 cam 转为 uint8 类型，范围为 [0, 255]
    cam = np.uint8(255 * cam)

    # 创建热力图
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # 确保 heatmap 和 img 的大小一致
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 叠加热力图和原始图像
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # 显示叠加图像
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 如果提供了 output_path，则保存叠加图像
    if output_path:
        cv2.imwrite(output_path, overlay)

def main(cam_npy_path, img_path, output_path=None):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return

    # 读取 CAM 数据
    cam_data = np.load(cam_npy_path, allow_pickle=True).item()
    cam = cam_data.get("attn_highres")  # 从 npy 文件中获取 'attn_highres'

    if cam is None:
        print(f"Error: 'attn_highres' not found in {cam_npy_path}")
        return

    # 确保 cam 是一个三维数组（有多个热力图）
    visualize_cam_on_image(img, cam[0], output_path)

if __name__ == "__main__":
    cam_npy_dir = "/home/zhang.13617/Desktop/BioCLIP_visualize/output/cams"
    train_absolute_txt = "/home/zhang.13617/Desktop/BioCLIP_visualize/iNat/train_absolute.txt"
    output_dir = "/home/zhang.13617/Desktop/BioCLIP_visualize/output/cams"
    
    # 获取 .npy 文件的路径
    cam_npy_paths = sorted([os.path.join(cam_npy_dir, f) for f in os.listdir(cam_npy_dir) if f.endswith('.npy')])

    # 读取 train_absolute.txt 中的图像绝对路径
    with open(train_absolute_txt, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]

    # 创建基于文件名的映射字典
    cam_dict = {os.path.basename(npy_path).replace('.npy', ''): npy_path for npy_path in cam_npy_paths}

    # 遍历图像路径并匹配 .npy 文件
    for img_path in img_paths:
        base_name = os.path.basename(img_path).replace('.jpg', '')
        cam_npy_path = cam_dict.get(base_name)
        
        if cam_npy_path:
            output_filename = base_name + '_output.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            # 调用 main 函数
            main(cam_npy_path, img_path, output_path)
        else:
            print(f"Error: No corresponding .npy file found for {img_path}")