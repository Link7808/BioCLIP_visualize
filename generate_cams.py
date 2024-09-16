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
sys.path.insert(0, '/home/zhang.13617/Desktop/CLIP-ES')
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

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])

    return dataset_list

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


def perform(process_id, dataset_list, args, model, fg_text_features, cam):
    device_id = "cuda:1"
    databin = dataset_list[process_id]
    model = model.to(device_id)
    fg_text_features = fg_text_features.to(device_id)
    for im_idx, im in enumerate(tqdm(databin)):
        img_path = os.path.join(args.img_root, im)
        
        image = Image.open(img_path).convert("RGB")  
        ori_width, ori_height = image.size 

        label_list = class_names  
        label_id_list = list(range(len(class_names))) 
        
        if len(label_list) == 0:
            print(f"{im} has no valid class names.")
            continue

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

            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            input_tensor = [image_features, fg_features_temp.to(device_id), h, w]

            for idx, label in enumerate(label_list):
                keys.append(class_names.index(label))
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

            keys = torch.tensor(keys)
            #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))


        #cam_all_scales = cam_all_scales[0]
        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]

        np.save(os.path.join(args.cam_out_dir, im.replace('jpg', 'npy')),
                {"keys": keys.numpy(),
                # "strided_cam": cam_per_scales.cpu().numpy(),
                #"highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
                "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
                })
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_root', type=str, default='/home/xxx/datasets/VOC2012/JPEGImages')
    parser.add_argument('--split_file', type=str, default='./voc12/train.txt')
    parser.add_argument('--cam_out_dir', type=str, default='./final/ablation/baseline')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" 
    print(device)

    train_list = np.loadtxt(args.split_file, dtype=str)
    train_list = [x + '.jpg' for x in train_list]
    train_list = train_list[:3]  


    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    fg_text_features = zeroshot_classifier(class_names, ['a clean origami {}.'], model)

    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, fg_text_features, cam)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(dataset_list, args, model, fg_text_features, cam))
