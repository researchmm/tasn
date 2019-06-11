# TASN
Code (MXNet version) for our cvpr'19 paper "Looking for the Devil in the Details: Learning Trilinear Attention Sampling Network for Fine-grained Image Recognition"

![alt text](https://user-images.githubusercontent.com/35843017/59253558-14314380-8c61-11e9-9c90-c8de5442ccad.jpg)

Prerequisites
-------
cuda version = 8.0
cudnn5.0
nccl
libopenblas
liblapack
libopencv 

Install
-------
First clone this repository:

    sudo git clone https://github.com/Heliang-Zheng/TASN.git
    cd TASN/tasn-mxnet

Then, please follow https://mxnet.incubator.apache.org/install/build_from_source.html to compile and install mxnet.

Or download pre-build mxnet (with cuda 8.0): https://drive.google.com/open?id=1Sfpw0x5XLqBFWAt99-zKOp4jAbOxm5Ws and install by:

    cd TASN/tasn-mxnet/example/tasn
    sudo bash install.sh


Train TASN
-------
1) get into the tasn dir:

        cd TASN/tasn-mxnet/example/tasn

2) download data and pretrained model (on ImageNet):

        sudo bash init.sh

3) set your nccl path in train.sh

4) run :

        sudo bash train.sh

Experiments settings:
on CUB-200-2011 dataset : http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

CNN input resolution: 224*224

Accuracy: 87.0%

Just changing the scale of AttSampler() in train.py from 224/512 to 336/512 to obtain the accuracy of 88.0%


Model:
-------
cub_224_87 https://drive.google.com/open?id=1uw9MVNVZqBTppN4TBbHB10CxoQonsTx9

cub_336_88 https://drive.google.com/open?id=1qQo8o2C5JpwxJGhrfk2xHM-f6kpxDKd1


Added files:
-------
example/tasn/*

src/operator/contrib/att_sampler-inl.h

src/operator/contrib/att_sampler.cc

src/operator/contrib/att_sampler.cu

PyTorch versioin
-------
TODO
