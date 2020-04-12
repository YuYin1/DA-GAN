# Dual-Attention GAN for Large-Pose Face Frontalization (DA-GAN)

This repository contains the code for our paper [Dual-Attention GAN for Large-Pose Face Frontalization](https://arxiv.org/abs/2002.07227) (**FG2020**).

## Train
Change data directory in option.py.
For training, run 
`python main.py --save_results --save_gt --save_models`.

## Test
For testing, run
`python main.py --save test_folder --test_only --save_results --save_gt --pre_train ../experiment/model.pth`.


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

