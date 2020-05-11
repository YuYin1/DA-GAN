# Dual-Attention GAN for Large-Pose Face Frontalization (DA-GAN)

This repository contains the code for our paper [Dual-Attention GAN for Large-Pose Face Frontalization](https://arxiv.org/abs/2002.07227) (**FG2020**).

## Requirements
The code is tested on:
- Python 3.6+
- Pytorch 0.4.1

## Data
We provide training and testing code for [MultiPIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html). Faces are first cropped and save to the folder with the structure as follows:
- MultiPIE/cropped
	- gallery
	- front
	- pose
	- mask_hair_ele_face

As for the face parser model, we use [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). 3 segments (i.e., hair, keypoints, and face) are generated and saved in the folder of `mask_hair_ele_face`.

The original CAS-PEAL-R1 dataset could be found at: [CAS-PEAL-R1](http://www.jdl.ac.cn/peal/). The cropped one could be downloaded from [here](https://drive.google.com/drive/folders/1OLTGh15CuhyRXA0nOXVnKKjbYAg9UOLE?usp=sharing).

## Train
Change data directory in option.py.
For training, run 
`python main.py --save_results --save_gt --save_models`.

We include an identity loss in our code, which is refered to [LightCNN](https://github.com/AlfredXiangWu/LightCNN#datasets). The pretrained LightCNN model can be downloaded from https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view. After downloading, save the model under the folder `src`.

## Test
For testing, run
`python main.py --save test_folder --test_only --save_results --save_gt --pre_train ../experiment/model.pth`.

As for the face parser model, we use https://github.com/zllrunning/face-parsing.PyTorch. 


## Citation
Please cite this paper in your publications if it helps your research:

>@article{yin2020dualattention,
    title={Dual-Attention GAN for Large-Pose Face Frontalization},
    author={Yu Yin and Songyao Jiang and Joseph P. Robinson and Yun Fu},
    year={2020},
    eprint={2002.07227},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

## Acknowledgement
we refer to [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) for the framework of training code. The self-attention module is brought from [SAGAN](https://github.com/heykeetae/Self-Attention-GAN). The identity loss module is brought from [LightCNN](https://github.com/AlfredXiangWu/LightCNN#datasets).