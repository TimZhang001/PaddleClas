# MixNet 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
      - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
- [4. 模型推理部署](#4)
  - [4.1 推理模型准备](#4.1)
  - [4.2 基于 Python 预测引擎推理](#4.2)
  - [4.3 基于 C++ 预测引擎推理](#4.3)
  - [4.4 服务化部署](#4.4)
  - [4.5 端侧部署](#4.5)
  - [4.6 Paddle2ONNX 模型转换与预测](#4.6)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

MixNet 是谷歌出的一篇关于轻量级网络的文章，主要工作就在于探索不同大小的卷积核的组合。作者发现目前网络有以下两个问题：

- 小的卷积核感受野小，参数少，但是准确率不高
- 大的卷积核感受野大，准确率相对略高，但是参数也相对增加了很多

为了解决上面两个问题，文中提出一种新的混合深度分离卷积(MDConv)(mixed depthwise convolution)，将不同的核大小混合在一个卷积运算中，并且基于 AutoML 的搜索空间，提出了一系列的网络叫做 MixNets，在 ImageNet 上取得了较好的效果。[论文地址](https://arxiv.org/pdf/1907.09595.pdf)

<a name='1.2'></a>

### 1.2 模型指标

| Models | Top1 | Top5 | Reference<br>top1| Reference<br>top5 | FLOPS<br>(M) | Params<br/>(M) |
|:--:|:--:|:--:|:--:|:--:|----|
| MixNet_S | 76.28 | 92.99 | 75.8 | - | 252.977 | 4.167 |
| MixNet_M | 77.67 | 93.64 | 77.0 | - | 357.119 | 5.065 |
| MixNet_L | 78.60 | 94.37 | 78.9 | - | 579.017 | 7.384 |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models   | Crop Size | Resize Short Size | FP32<br/>Batch Size=1<br/>(ms) | FP32<br/>Batch Size=4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
| -------- | --------- | ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| MixNet_S | 224       | 256               | 2.31                           | 3.63                           | 5.20                           |
| MixNet_M | 224       | 256               | 2.84                           | 4.60                           | 6.62                           |
| MixNet_L | 224       | 256               | 3.16                           | 5.55                           | 8.03                           |

关于 Inference speed 等信息，敬请期待。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测


此部分内容包括训练环境配置、ImageNet数据的准备、SwinTransformer 在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/SwinTransformer/` 中提供了 SwinTransformer 的训练配置，可以通过如下脚本启动训练：此部分内容可以参考[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

**备注：** 由于 SwinTransformer 系列模型默认使用的 GPU 数量为 8 个，所以在训练时，需要指定8个GPU，如`python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c xxx.yaml`, 如果使用 4 个 GPU 训练，默认学习率需要减小一半，精度可能有损。


<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理) 对 SwinTransformer 完成推理预测。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../inference_deployment/cpp_deploy_on_windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](@shuilong)来完成相应的部署工作。