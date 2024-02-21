### RF++

My extensions to Zhouxia Wang et al.'s [RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus).
Notable changes:
- 2x faster and 40% memory usage when training ROHQD (see [optimizations](dec_14_2023_optimization_ablation.md)
- Updated library requirements
- more config options (when training ROHQD)
- Pytorch profiler support
- A collection of CUDA kernels for NHWC Group Norm which outperform both Pytorch's native GN NCHW kernels AND stable-fast's Triton NHWC kernels
  - 40% training speedup + additional 30% memory savings (so 2.8x faster training, 28% memory usage from original RF++)

## Environment

- python>=3.9
- pytorch>=2.2
- pytorch-lightning==2.1.2
- omegaconf>=2.0.0
- basicsr>=1.4.2
- realesrgan==0.3.0
- pillow-simd>=7.0.0
- tensorboard>=2.0.0

**Warning** Different versions of pytorch-lightning and omegaconf may lead to errors or different results.

## Preparations of dataset and models

**Dataset**: 
- Training data: Both **ROHQD** and **RestoreFormer++** in our work are trained with **FFHQ** which attained from [FFHQ repository](https://github.com/NVlabs/ffhq-dataset). The original size of the images in FFHQ are 1024x1024. We resize them to 512x512 with bilinear interpolation in our work. Link this dataset to ./data/FFHQ/image512x512.
- <a id="testset">Test data</a>: [CelebA-Test](https://pan.baidu.com/s/1iUvBBFMkjgPcWrhZlZY2og?pwd=test), [LFW-Test](http://vis-www.cs.umass.edu/lfw/#views), [WebPhoto-Test](https://xinntao.github.io/projects/gfpgan), and [CelebChild-Test](https://xinntao.github.io/projects/gfpgan)

**Model**: 
Both pretrained models used for training and the trained model of our RestoreFormer and RestoreFormer++ can be attained from [Google Driver](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wzhoux_connect_hku_hk/EkZhGsLBtONKsLlWRmf6g7AB_VOA_6XAKmYUXLGKuNBsHQ?e=ic2LPl). Link these models to ./experiments.

<!-- ## <a id="metrics">Metrics</a> -->
<h2 id="inference">Quick Inference</h2>

    python inference.py -i data/aligned -o results/RF++/aligned -v RestoreFormer++ -s 2 --aligned --save
    python inference.py -i data/raw -o results/RF++/raw -v RestoreFormer++ -s 2 --save
    python inference.py -i data/aligned -o results/RF/aligned -v RestoreFormer -s 2 --aligned --save
    python inference.py -i data/raw -o results/RF/raw -v RestoreFormer -s 2 --save

**Note**: Related codes are borrowed from [GFPGAN](https://github.com/TencentARC/GFPGAN). 

## Test
    sh scripts/test.sh

- This codebase is available for both **RestoreFormer** and **RestoreFormerPlusPlus**. Determinate the specific model with *exp_name*.
- Setting the model path with *root_path*
- Restored results are save in *out_root_path*
- Put the degraded face images in *test_path*
- If the degraded face images are aligned, set *--aligned*, else remove it from the script. The provided test images in data/aligned are aligned, while images in data/raw are unaligned and contain several faces.

## Training
    sh scripts/run.sh

- This codebase is available for both **RestoreFormer** and **RestoreFormerPlusPlus**. Determinate the training model with *conf_name*. 'HQ_Dictionary' and 'RestoreFormer' are for **RestoreFormer**, while 'ROHQD' and 'RestoreFormerPlusPlus' are for **RestoreFormerPlusPlus**.
- While training 'RestoreFormer' or 'RestoreFormerPlusPlus', *'ckpt_path'* in the corresponding configure files in configs/ sholud be updated with the path of the trained model of 'HQ_Dictionary' or 'ROHQD'.

<!-- ## <a id="metrics">Metrics</a> -->
## Metrics
    sh scripts/metrics/run.sh
