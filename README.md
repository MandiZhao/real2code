# Real2code: Reconstruct Articulated Objects via Code Generation
**[Mandi Zhao](https://mandizhao.github.io/), [Yijia Weng](https://yijiaweng.github.io/), [Dominik Bauer](https://dornik.github.io/), [Shuran Song](https://shurans.github.io/)**

[Arxiv](https://arxiv.org/abs/2406.08474) | [Website](https://real2code.github.io/)

 
<img src="real2code-teaser.jpg" alt="teaser" width="800"/>

## Installation
Use conda environment with Python 3.9, and install packages from the provided `.yaml` file
```
conda create -n real2code python=3.9
conda activate real2code
conda env update --file environment.yml --prune
```

## Code Overview
This repo encapsulates multiple sub-modules of the Real2Code pipeline. 

- **Dataset**: Overall, all modules use the same synthetic dataset of RGBD images, part-level meshes, and code snippets for joint structures for each object. We have released this dataset [here](https://drive.google.com/drive/folders/1rkUP7NBRQX5h6ixJr9SvX0Vh3fhj1YqO?usp=drive_link), and provide processing & rendering utility scripts in `data_utils/` if you want to generate your own data. 
- **Part-level 2D-Segmentation and 3D Shape Completion**: With the same set of objects, we fine-tune a 2D SAM model for part-level segmentation and train a PointNet-based model for 3D shape completion. More details on each sub-module is further documented in the READMEs in [part segmentation](https://github.com/MandiZhao/real2code/tree/main/part_segmentation) and [shape completion](https://github.com/MandiZhao/real2code/tree/main/shape_complete). 
- **LLM Fine-tuning**: We fine-tune a CodeLlama model on the code representations of our articulated objects. See [this fork](https://github.com/MandiZhao/open_flamingo) for LLM fine-tuning script. 
- **Real World Evaluation** See `real_obj/`. We use [DUSt3R](https://github.com/naver/dust3r) to achieve reconstruction from multi-view pose-free RGB images, the DUSt3R-generated 3D pointmaps are provided in the real world dataset below. 

## Dataset 
### Synthetic Data
Our dataset is built on top of PartNet-Mobilty assets, and the same set of objects are used for training and testing throughout our SAM fine-tuning, shape completion model training, and LLM fine-tuning modules. The full dataset will be released here: https://drive.google.com/drive/folders/1rkUP7NBRQX5h6ixJr9SvX0Vh3fhj1YqO?usp=drive_link 

### Real-world Objects
We have released the real objects data used for evaluating Real2Code. These are objects found in the common lab/household settings around Stanford campus. Raw data is captured using a LiDAR-equipped iPhone camera and the [3dScanner App](https://apps.apple.com/us/app/3d-scanner-app/id1419913995)
- Download: [Google Drive Link](https://drive.google.com/drive/folders/1LSjpatlAzTXxOUcwbGjZR_ST7aeUEjn2?usp=drive_link)
- Structure: each object folder is structured as follows:
  ```
  ls obj_id/
  - raw/
  - sam/
  - a list of (id.jpg, id_mask.png, id_scene.npz),
  ```
  Each `id` corresponds to one 512x512 RGB image selected from the raw dataset, e.g. `00000.jpg`; `id_mask.png` is the foreground object mask obtained from prompting the SAM model with randomly sampled query points in the image margin area; `id_scene.npz` is the globally-aligned 3D point-cloud obtained from [DUSt3R](https://github.com/naver/dust3r). 

