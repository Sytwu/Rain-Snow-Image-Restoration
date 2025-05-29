# NYCU Computer Vision 2025 Spring HW4
StudentID: 111550159 \
Name: Li-Zhong Szu-Tu (司徒立中)

## Introduction
In this task, the dataset is designed for image restoration, consisting of 1600 paired training images for each degradation type (rain and snow), where each pair includes a degraded image and its corresponding clean image. The test set contains 100 images with unknown degradation types (either rain or snow). The objective is to perform image restoration by predicting clean images from degraded inputs, effectively handling both deraining and desnowing tasks.\
\
The performance is evaluated solely using Peak Signal-to-Noise Ratio (PSNR). PSNR is a widely used metric that measures the quality of reconstructed images by comparing them to their ground-truth clean versions, calculated as the ratio between the maximum possible pixel value and the mean squared error between the restored and clean images. Higher PSNR values indicate better image restoration quality.\
\
Additionally, the strategies for this task are subject to specific constraints. First, no external data is allowed, ensuring the focus remains on model architecture design rather than data augmentation or collection. Second, no pre-trained weights can be used, meaning the model must be trained from scratch. The chosen model for this task is PromptIR, which is capable of handling multiple degradation effects (rain and snow) simultaneously within a unified framework.\
\
The dataset can be downloaded [Here](https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul?usp=share_link)!

## Code Reliability
utils/dataset_utils.py \
\
options.py \
\
train.py \
\
demo.py \
\
![image](https://github.com/user-attachments/assets/c541fc09-0ec5-4c1b-bbfa-a5007fd23cf1) \
(The code only includes what I have modified, rather than the full content.)


## How to install
How to install dependences
```
conda env create -f environment.yml
conda activate promptir
```

## How to run
How to execute the code
```
# Training
python train.py --de_type denoise derain desnow --epochs 100 --patch_size 256 --num_gpu 2

# Inference
python demo.py --test_path './test/demo/' --output_path './output/demo/' --ckpt_path 'best.ckpt'

```
My model weights can be downloaded [Here](https://drive.google.com/drive/folders/1wz4kqwkiQP1b7oARfwDfmJtsvmnf5ekY?usp=sharing)!

## Performance snapshot
A shapshot of the leaderboard
Please note that my student ID is 111550159, there's a typo on the leaderboard.
![喔不](https://github.com/user-attachments/assets/9f9a4595-00bd-4482-bdad-c394e2b28aaa)

\
Last Update: 2025/05/29 12:21 a.m. (GMT+8)
