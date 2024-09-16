

## Reqirements

```
# create conda env
conda create -n clip-es python=3.9
conda activate clip-es

# install packages
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python ftfy regex tqdm ttach tensorboard lxml cython

# install pydensecrf from source
git clone https://github.com/lucasb-eyer/pydensecrf
cd pydensecrf
python setup.py install
```




#
## Usage
### Step 1. Generate CAMs for train (train_aug) set.
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
--img_root /home/zhang.13617/Desktop/CLIP-ES/datasets/Birds \
--split_file ./Bird1/train.txt \
--num_workers 1 \
--cam_out_dir ./output/Bird1/cams



### Step 2. Evaluate generated CAMs and use CRF to postprocess
```
# (optional) evaluate generated CAMs
## for VOC12
python eval_cam.py --cam_out_dir ./output/voc12/cams --cam_type attn_highres --gt_root /your_home_dir/datasets/VOC2012/SegmentationClassAug --split_file ./voc12/train.txt
## for COCO14
python eval_cam.py --cam_out_dir ./output/coco14/cams --cam_type attn_highres --gt_root /your_home_dir/datasets/COCO2014/SegmentationClass --split_file ./coco14/train.txt

# use CRF process to generate pseudo masks 
(realize confidence-guided loss by setting pixels with low confidence to 255)
## for VOC12 
python eval_cam_with_crf.py --cam_out_dir ./output/voc12/cams --gt_root /your_home_dir/datasets/VOC2012/SegmentationClassAug --image_root /your_home_dir/datasets/VOC2012/JPEGImages --split_file ./voc12/train_aug.txt --pseudo_mask_save_path ./output/voc12/pseudo_masks
## for COCO14
python eval_cam_with_crf.py --cam_out_dir ./output/coco14/cams --gt_root /your_home_dir/datasets/COCO2014/SegmentationClass --image_root /your_home_dir/datasets/COCO2014/JPEGImages/train2014 --split_file ./coco14/train.txt --pseudo_mask_save_path ./output/coco2014/pseudo_masks

# eval CRF processed pseudo masks
## for VOC12 
python eval_cam_with_crf.py --cam_out_dir ./output/voc12/cams --gt_root /your_home_dir/datasets/VOC2012/SegmentationClassAug --image_root /your_home_dir/datasets/VOC2012/JPEGImages --split_file ./voc12/train_aug.txt --eval_only
## for COCO14
python eval_cam_with_crf.py --cam_out_dir ./output/coco14/cams --gt_root /your_home_dir/datasets/COCO2014/SegmentationClass --image_root /your_home_dir/datasets/COCO2014/JPEGImages/train2014 --split_file ./coco14/train.txt --eval_only

```


## Acknowledgement
We borrowed the code from [CLIP](https://github.com/openai/CLIP) and [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a). Thanks for their wonderful works.

Code is modified from [CLIP-ES](https://github.com/linyq2117/CLIP-ES) and [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a)
## Citation


```bibtex
@InProceedings{Lin_2023_CVPR,
    author    = {Lin, Yuqi and Chen, Minghao and Wang, Wenxiao and Wu, Boxi and Li, Ke and Lin, Binbin and Liu, Haifeng and He, Xiaofei},
    title     = {CLIP Is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15305-15314}
}