# -*- coding:UTF-8 -*-
from pytorch_grad_cam import GradCAM
import torch
from PIL import Image
import numpy as np
import cv2
import os
import open_clip

from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import parse_xml_to_dict, scoremap2bbox
from clip_text import class_names#, imagenet_templates
import argparse
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

import types
import sys
sys.path.insert(0, '/home/zhang.13617/Desktop/BioCLIP_visualize')
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")

def reshape_transform(tensor, height=None, width=None):
    tensor = tensor.permute(1, 0, 2)  # Shape: (batch_size, seq_len, hidden_dim)
    seq_len = tensor.size(1)
    batch_size = tensor.size(0)
    hidden_dim = tensor.size(2)
    n_tokens = seq_len - 1  # Exclude the first token (e.g., [CLS] token)

    if height is not None and width is not None:
        if height * width != n_tokens:
            raise ValueError(f"Provided height ({height}) and width ({width}) do not multiply to n_tokens ({n_tokens})")
    elif height is not None:
        if n_tokens % height != 0:
            raise ValueError(f"Cannot reshape: n_tokens ({n_tokens}) is not divisible by provided height ({height})")
        width = n_tokens // height
    elif width is not None:
        if n_tokens % width != 0:
            raise ValueError(f"Cannot reshape: n_tokens ({n_tokens}) is not divisible by provided width ({width})")
        height = n_tokens // width
    else:
        # Automatically determine height and width
        height = int(n_tokens ** 0.5)
        while n_tokens % height != 0 and height > 1:
            height -= 1
        if height == 1 and n_tokens % height != 0:
            raise ValueError(f"Cannot automatically determine height and width for n_tokens ({n_tokens})")
        width = n_tokens // height

    result = tensor[:, 1:, :].reshape(batch_size, height, width, hidden_dim)
    result = result.permute(0, 3, 1, 2)  # Shape: (batch_size, hidden_dim, height, width)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs

def read_top_classes(file_path):
    image_top_classes = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith('Image:'):
                image_name = line.split(' ')[1].strip()
                # Read the next two lines for Top 1 and Top 2 classes
                top1_line = lines[idx + 1].strip()
                top2_line = lines[idx + 2].strip()

                # Extract class names
                top1_class = top1_line.split(':')[1].split('with')[0].strip()
                top2_class = top2_line.split(':')[1].split('with')[0].strip()

                image_top_classes[image_name] = [top1_class, top2_class]
                idx += 3
            else:
                idx += 1
    return image_top_classes
def perform(dataset_list, args, model, fg_text_features, cam):
    device_id = "cuda:1"
    model = model.to(device_id)
    fg_text_features = fg_text_features.to(device_id)
    image_top_classes = read_top_classes('/home/zhang.13617/Desktop/BioCLIP_visualize/iNat/output.txt')#

    for im_idx, img_path in enumerate(tqdm(dataset_list)):

        image_filename = os.path.basename(img_path)#111
        top_classes = image_top_classes.get(image_filename, [])#111
        if not top_classes:
            print(f"No top classes found for image {image_filename}. Skipping.")
            continue


        image = Image.open(img_path).convert("RGB")  
        ori_width, ori_height = image.size 

        label_list = top_classes
        label_id_list = list(range(len(class_names))) 

        ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
        ms_imgs = [ms_imgs[0]]  
        cam_all_scales = []
        highres_cam_all_scales = []
        refined_cam_all_scales = []

        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(device_id)
            image_features, attn_weight_list = model.encode_image(image, h, w)


            cam_to_save = []
            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []
            refined_cam_list = []
            class_labels = []




            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            input_tensor = [image_features, fg_features_temp.to(device_id), h, w]

            for idx, label in enumerate(label_list):
                class_labels.append(label)
                keys.append(class_names.index(label))
                class_index = class_names.index(label)
                targets = [ClipOutputTarget(label_list.index(label))]

                #torch.cuda.empty_cache()
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                                   targets=targets,
                                                                                   target_size=None)  

                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

                if idx == 0:
                    attn_weight_list.append(attn_weight_last)
                    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                    attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                    attn_weight = torch.mean(attn_weight, dim=0)
                    attn_weight = attn_weight[0].cpu().detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

                for _ in range(2):
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
                trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam)
                cam_to_refine = cam_to_refine.view(-1,1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
                cam_refined = cam_refined.cpu().numpy().astype(np.float32)
                cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
                refined_cam_to_save.append(torch.tensor(cam_refined_highres))
                output_filename = os.path.basename(img_path).replace('.jpg', f'_{label}.npy')
                refined_cam_list.append(cam_refined_highres)
                np.save(os.path.join(args.cam_out_dir, output_filename),
                       {"key": class_index,
                        "attn_highres": cam_refined_highres.astype(np.float16),
                         })

            #keys = torch.tensor(keys)
            #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            #highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))
          #  refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))


        #cam_all_scales = cam_all_scales[0]
  
        # 提取文件名并替换扩展名为 .npy
        #output_filename = os.path.basename(img_path).replace('.jpg', '.npy')
        #if len(refined_cam_list) == 2:
            # Compute the difference between the two CAMs
          #  cam_diff = refined_cam_list[0] - refined_cam_list[1]

            # Normalize the difference map for visualization
           # cam_diff_norm = (cam_diff - np.min(cam_diff)) / (np.max(cam_diff) - np.min(cam_diff) + 1e-8)

            # Save the difference map
          #  output_diff_filename = os.path.basename(img_path).replace('.jpg', f'_{class_labels[0]}_minus_{class_labels[1]}.npy')
           # np.save(os.path.join(args.cam_out_dir, output_diff_filename),
                  #  {"key_diff": (class_labels[0], class_labels[1]),
                   # "cam_diff": cam_diff_norm.astype(np.float16),
#})
       # else:
           # print(f"Expected two refined CAMs, but got {len(refined_cam_list)} for image {img_path}")

        # 使用 os.path.join 拼接保存路径
        #np.save(os.path.join(args.cam_out_dir, output_filename),
                #{"keys": keys.numpy(),
                # "strided_cam": cam_per_scales.cpu().numpy(),
                #"highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
               # "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
              #  })
   # return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split_file', type=str, default='/home/zhang.13617/Desktop/BioCLIP_visualize/iNat/train_absolute.txt')
    parser.add_argument('--cam_out_dir', type=str, default='/home/zhang.13617/Desktop/BioCLIP_visualize/output/cams_top')
    args = parser.parse_args()

    device = "cuda" 
    print(device)

    train_list = np.loadtxt(args.split_file, dtype=str)


    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    model, preprocess, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    fg_text_features = zeroshot_classifier(class_names, ['a clean origami {}.'], model)

    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    perform(train_list, args, model, fg_text_features, cam) 
