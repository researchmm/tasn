
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


Original MXNet readme
------

<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<div align="center">
  <a href="https://mxnet.incubator.apache.org/"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png"></a><br>
</div>

Apache MXNet (incubating) for Deep Learning
=====
| Master         | Docs          | License  |
| :-------------:|:-------------:|:--------:|
| [![Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/badge/icon)](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/)  | [![Documentation Status](http://jenkins.mxnet-ci.amazon-ml.com/job/restricted-website-build/badge/icon)](https://mxnet.incubator.apache.org/) | [![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE) |

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

Apache MXNet (incubating) is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
[blue prints and guidelines](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts) for building
deep learning systems, and interesting insights of DL systems for hackers.

Ask Questions
-------------
* Please use [discuss.mxnet.io](https://discuss.mxnet.io/) for asking questions.
* Please use [mxnet/issues](https://github.com/apache/incubator-mxnet/issues) for reporting bugs.
* [Frequent Asked Questions](https://mxnet.incubator.apache.org/faq/faq.html)

How to Contribute
-----------------
* [Contribute to MXNet](https://mxnet.incubator.apache.org/community/contribute.html)

What's New
----------
* [Version 1.4.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.4.0) - MXNet 1.4.0 Release.
* [Version 1.3.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.3.1) - MXNet 1.3.1 Patch Release.
* [Version 1.3.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.3.0) - MXNet 1.3.0 Release.
* [Version 1.2.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.2.0) - MXNet 1.2.0 Release.
* [Version 1.1.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.1.0) - MXNet 1.1.0 Release.
* [Version 1.0.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0) - MXNet 1.0.0 Release.
* [Version 0.12.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.1) - MXNet 0.12.1 Patch Release.
* [Version 0.12.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.0) - MXNet 0.12.0 Release.
* [Version 0.11.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.11.0) - MXNet 0.11.0 Release.
* [Apache Incubator](http://incubator.apache.org/projects/mxnet.html) - We are now an Apache Incubator project.
* [Version 0.10.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.10.0) - MXNet 0.10.0 Release.
* [Version 0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [Version 0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [Version 0.8.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.8.0)
* [Updated Image Classification with new Pre-trained Models](./example/image-classification)
* [Notebooks How to Use MXNet](https://github.com/d2l-ai/d2l-en)
* [MKLDNN for Faster CPU Performance](./docs/tutorials/mkldnn/MKLDNN_README.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [Embedding Torch layers and functions in MXNet](https://mxnet.incubator.apache.org/faq/torch.html)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)
](https://github.com/dmlc/mxnet.js/)
* [Design Note: Design Efficient Deep Learning Data Loading Module](https://mxnet.incubator.apache.org/architecture/note_data_loading.html)
* [MXNet on Mobile Device](https://mxnet.incubator.apache.org/faq/smart_device.html)
* [Distributed Training](https://mxnet.incubator.apache.org/faq/multi_devices.html)
* [Guide to Creating New Operators (Layers)](https://mxnet.incubator.apache.org/faq/new_op.html)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)
* [Amalgamation and Go Binding for Predictors](https://github.com/jdeng/gomxnet/) - Outdated
* [Large Scale Image Classification](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification)

Contents
--------
* [Documentation](https://mxnet.incubator.apache.org/) and  [Tutorials](https://mxnet.incubator.apache.org/tutorials/)
* [Design Notes](https://mxnet.incubator.apache.org/architecture/index.html)
* [Code Examples](https://github.com/apache/incubator-mxnet/tree/master/example)
* [Installation](https://mxnet.incubator.apache.org/install/index.html)
* [Pretrained Models](http://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match imperative and symbolic programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for Python, Scala, C++, Java, Clojure, R and Julia
* Cloud-friendly and directly compatible with S3, HDFS, and Azure

License
-------
Licensed under an [Apache-2.0](https://github.com/apache/incubator-mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.

