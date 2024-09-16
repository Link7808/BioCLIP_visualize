

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




## Step 1. Generate CAMs for train (train_aug) set.
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
--img_root /home/zhang.13617/Desktop/CLIP-ES/datasets/Birds \
--split_file ./Bird1/train.txt \
--num_workers 1 \
--cam_out_dir ./output/Bird1/cams

```






## Acknowledgement

```
We borrowed the code from [CLIP](https://github.com/openai/CLIP) and [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a). Thanks for their wonderful works.

Code is modified from [CLIP-ES](https://github.com/linyq2117/CLIP-ES) and [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a)

```