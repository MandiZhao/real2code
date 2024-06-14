# Real2code: Reconstruct Articulated Objects via Code Generation
**[Mandi Zhao](https://mandizhao.github.io/), [Yijia Weng](https://yijiaweng.github.io/), [Dominik Bauer](https://dornik.github.io/), [Shuran Song](https://shurans.github.io/)**

[Arxiv](https://arxiv.org/abs/2406.08474) | [Website](https://real2code.github.io/)


## Installation
1. Use conda environment with Python 3.9:
```
conda create -n real2code python=3.9
```



## Code Overview
### Data Generation & Processing  
See `data_utils/`

### Kinematics-Aware SAM Fine-tuning 
See `image_segmentation/`

### Shape Completion 
See `shape_complete/`

### LLM fine-tuning
We use a custom fork of Open-Flamingo: https://github.com/mlfoundations/open_flamingo

### Real World Evaluation
See `real_obj/`. We use [DUSt3R](https://github.com/naver/dust3r) to achieve reconstruction from multi-view pose-free RGB images.

## Dataset 
### Synthetic Data
Our dataset is built on top of PartNet-Mobilty assets, and the same set of objects are used for training and testing throughout our SAM fine-tuning, shape completion model training, and LLM fine-tuning modules. 

### Real-world Objects
We have released the real objects data used for evaluating Real2Code. These are objects found in the common lab/household settings around Stanford campus. Raw data is captured using a LiDAR-equipped iPhone camera and the [3dScanner App](https://apps.apple.com/us/app/3d-scanner-app/id1419913995)
- Download: [Google Drive Link](https://drive.google.com/drive/folders/1LSjpatlAzTXxOUcwbGjZR_ST7aeUEjn2?usp=sharing)
- Structure: each object folder is structured as follows:
  ```
  ls obj_id/
  - raw/
  - sam/
  - a list of (id.jpg, id_mask.png, id_scene.npz),
  ```
  Each `id` corresponds to one 512x512 RGB image selected from the raw dataset, e.g. `00000.jpg`; `id_mask.png` is the foreground object mask obtained from prompting the SAM model with randomly sampled query points in the image margin area; `id_scene.npz` is the globally-aligned 3D point-cloud obtained from [DUSt3R](https://github.com/naver/dust3r). 

