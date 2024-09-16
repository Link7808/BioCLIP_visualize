import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def visualize_cam_on_image(img, cam, output_path=None):
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')  
    plt.show()

    if output_path:
        cv2.imwrite(output_path, overlay)

def main(cam_npy_path, img_path, output_path=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return

    cam_data = np.load(cam_npy_path, allow_pickle=True).item()
    cam = cam_data.get("attn_highres")  

    if cam is None:
        print(f"Error: 'attn_highres' not found in {cam_npy_path}")
        return

    visualize_cam_on_image(img, cam[0], output_path)

if __name__ == "__main__":

    cam_npy_dir = "/home/zhang.13617/Desktop/CLIP-ES/output/Bird1/cams/"
    img_dir = "/home/zhang.13617/Desktop/CLIP-ES/datasets/Birds/"
    output_dir = "/home/zhang.13617/Desktop/CLIP-ES/output/Bird1/cams/"
    
    cam_npy_paths = sorted([os.path.join(cam_npy_dir, f) for f in os.listdir(cam_npy_dir) if f.endswith('.npy')])
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    if len(cam_npy_paths) != len(img_paths):
        print("Error: The number of .npy files does not match the number of .jpg files.")
    else:
        for cam_npy_path, img_path in zip(cam_npy_paths, img_paths):
            output_filename = os.path.basename(img_path).replace('.jpg', '_output.jpg')
            output_path = os.path.join(output_dir, output_filename)
            
            main(cam_npy_path, img_path, output_path)