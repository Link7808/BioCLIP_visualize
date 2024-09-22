import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_cam_on_image(img, cam, output_path=None):
    # Normalize CAM
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Convert cam to uint8 type, range [0, 255]
    cam = np.uint8(255 * cam)

    # Create heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Ensure heatmap and img have the same size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Display overlay image
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Save overlay image if output_path is provided
    if output_path:
        cv2.imwrite(output_path, overlay)

def main(cam_npy_path, img_path, output_path=None):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return

    # Read CAM data
    try:
        cam_data = np.load(cam_npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading CAM npy file: {cam_npy_path}, error: {e}")
        return

    # Check if 'attn_highres' exists in the loaded npy data
    if "attn_highres" not in cam_data:
        print(f"Error: 'attn_highres' not found in {cam_npy_path}")
        return

    cam = cam_data.get("attn_highres")  # Get 'attn_highres' from npy file

    # Ensure 'cam' is valid
    if cam is None:
        print(f"Error: 'attn_highres' is None in {cam_npy_path}")
        return

    # If cam is a 3D array (multiple heatmaps), process each one
    if cam.ndim == 3:
        for idx in range(cam.shape[0]):
            cam_single = cam[idx]
            # Adjust output filename to include index
            output_path_idx = output_path.replace('.jpg', f'_{idx}.jpg')
            visualize_cam_on_image(img, cam_single, output_path_idx)
    else:
        visualize_cam_on_image(img, cam, output_path)

if __name__ == "__main__":
    cam_npy_dir = "/home/zhang.13617/Desktop/BioCLIP_visualize/output/cams"
    train_absolute_txt = "/home/zhang.13617/Desktop/BioCLIP_visualize/train/train_absolute.txt"
    output_dir = "/home/zhang.13617/Desktop/BioCLIP_visualize/output/cams"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get paths to .npy files sorted by filename
    cam_npy_paths = sorted([os.path.join(cam_npy_dir, f) for f in os.listdir(cam_npy_dir) if f.endswith('.npy')])

    # Read image absolute paths from train_absolute.txt
    with open(train_absolute_txt, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]

    # Check if the number of images and CAM files match
    if len(img_paths) != len(cam_npy_paths):
        print(f"Error: Number of images ({len(img_paths)}) does not match number of CAM files ({len(cam_npy_paths)}).")
    else:
        # Iterate over image paths and CAM npy paths in order
        for img_path, cam_npy_path in zip(img_paths, cam_npy_paths):
            base_name = os.path.basename(img_path).replace('.jpg', '')
            
            # Extract class label or index from CAM filename
            cam_filename = os.path.basename(cam_npy_path)
            class_label = cam_filename.replace('.npy', '').split('_')[-1]
            
            output_filename = base_name + f'_{class_label}_output.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            # Call main function
            main(cam_npy_path, img_path, output_path)