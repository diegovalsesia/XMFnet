[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-learning-for-image-guided-point/point-cloud-completion-on-shapenet-vipc)](https://paperswithcode.com/sota/point-cloud-completion-on-shapenet-vipc?p=cross-modal-learning-for-image-guided-point)

# XMFnet : Cross-Modal Learning for Image-Guided Point Cloud Shape Completion
This repository contains the official implementation for "Cross-modal Learning for Image-Guided Point Cloud Shape Completion" (NeurIPS 2022) [paper](https://arxiv.org/pdf/2209.09552.pdf)

![](figs/mmpc_arch.png)

## Requirements
The code has been developed with the following dependecies:

- Python 3.8 
- CUDA version 10.2
- G++ or GCC 7.5.0
- Pytorch 1.10.2

To setup the environment and install all the required packages run:

```setup
sh setup.sh
```

It automatically creates the environment and install all the required packages.

If something goes wrong please consider to follow the steps in setup manually.



## Dataset 

The dataset is borrowed from ["View-guided point cloud completion"](https://github.com/Hydrogenion/ViPC).

First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run ``cat ShapeNetViPC-Dataset.tar.gz* | tar zx``, you will get ``ShapeNetViPC-Dataset`` contains three floders: ``ShapeNetViPC-Partial``, ``ShapeNetViPC-GT`` and ``ShapeNetViPC-View``. 

For each object, the dataset include partial point cloud (``ShapeNetViPC-Patial``), complete point cloud (``ShapeNetViPC-GT``) and corresponding images (``ShapeNetViPC-View``) from 24 different views. You can find the detail of 24 cameras view in ``/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt``.


## Training
The file config.py contains the configuration for all the training parameters.

To train the models in the paper, run this command:

```train
python train.py 
```


## Evaluation

To evaluate the models (select the specific category in config.py):

```eval
python eval.py 
```


## Pre-trained Models

You can download pretrained models here:


## Results

# [Point Cloud Completion on ShapeNet-ViPC](https://paperswithcode.com/sota/point-cloud-completion-on-shapenet-vipc)

![](figs/res_2.png)

## Acknowledgments
Some of the code is borrowed from [AXform](https://github.com/kaiyizhang/AXform)


## Citation
If you find our work useful in your research, please consider citing: 

```
@misc{https://doi.org/10.48550/arxiv.2209.09552,
  doi = {10.48550/ARXIV.2209.09552},
  url = {https://arxiv.org/abs/2209.09552},
  author = {Aiello, Emanuele and Valsesia, Diego and Magli, Enrico},
  title = {Cross-modal Learning for Image-Guided Point Cloud Shape Completion},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```


## License 
Our code is released under MIT License (see LICENSE file for details).



