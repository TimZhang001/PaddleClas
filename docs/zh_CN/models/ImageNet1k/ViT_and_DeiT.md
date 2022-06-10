# ViT 与 DeiT 系列
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

ViT（Vision Transformer）系列模型是 Google 在 2020 年提出的，该模型仅使用标准的 Transformer 结构，完全抛弃了卷积结构，将图像拆分为多个 patch 后再输入到 Transformer 中，展示了 Transformer 在 CV 领域的潜力。[论文地址](https://arxiv.org/abs/2010.11929)。

DeiT（Data-efficient Image Transformers）系列模型是由 FaceBook 在 2020 年底提出的，针对 ViT 模型需要大规模数据集训练的问题进行了改进，最终在 ImageNet 上取得了 83.1%的 Top1 精度。并且使用卷积模型作为教师模型，针对该模型进行知识蒸馏，在 ImageNet 数据集上可以达到 85.2% 的 Top1 精度。[论文地址](https://arxiv.org/abs/2012.12877)。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ViT_small_patch16_224 | 0.7553 | 0.9211 | 0.7785 | 0.9342 | 9.41 | 48.60 |
| ViT_base_patch16_224  | 0.8187 | 0.9618 | 0.8178 | 0.9613 | 16.85 | 86.42 |
| ViT_base_patch16_384  | 0.8414 | 0.9717 | 0.8420 | 0.9722 | 49.35 | 86.42 |
| ViT_base_patch32_384  | 0.8176 | 0.9613 | 0.8166 | 0.9613 | 12.66 | 88.19 |
| ViT_large_patch16_224 | 0.8303 | 0.9655 | 0.8306 | 0.9644 | 59.65 | 304.12 |
| ViT_large_patch16_384 | 0.8513 | 0.9736 | 0.8517 | 0.9736 | 174.70 | 304.12 |
| ViT_large_patch32_384 | 0.8153 | 0.9608 | 0.815  | -      | 44.24 | 306.48 |


| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeiT_tiny_patch16_224            | 0.7208 | 0.9112 | 0.722 | 0.911 | 1.07 | 5.68 |
| DeiT_small_patch16_224           | 0.7982 | 0.9495 | 0.799 | 0.950 | 4.24 | 21.97 |
| DeiT_base_patch16_224            | 0.8180 | 0.9558 | 0.818 | 0.956 | 16.85 | 86.42 |
| DeiT_base_patch16_384            | 0.8289 | 0.9624 | 0.829 | 0.972 | 49.35 | 86.42 |
| DeiT_tiny_distilled_patch16_224  | 0.7449 | 0.9192 | 0.745 | 0.919 | 1.08 | 5.87 |
| DeiT_small_distilled_patch16_224 | 0.8117 | 0.9538 | 0.812 | 0.954 | 4.26 | 22.36 |
| DeiT_base_distilled_patch16_224  | 0.8330 | 0.9647 | 0.834 | 0.965 | 16.93 | 87.18 |
| DeiT_base_distilled_patch16_384  | 0.8520 | 0.9720 | 0.852 | 0.972 | 49.43 | 87.18 |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models                     | Crop Size | Resize Short Size | FP32<br/>Batch Size=1<br/>(ms) | FP32<br/>Batch Size=4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
| -------------------------- | --------- | ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| ViT_small_<br/>patch16_224 | 256       | 224               | 3.71                           | 9.05                           | 16.72                          |
| ViT_base_<br/>patch16_224  | 256       | 224               | 6.12                           | 14.84                          | 28.51                          |
| ViT_base_<br/>patch16_384  | 384       | 384               | 14.15                          | 48.38                          | 95.06                          |
| ViT_base_<br/>patch32_384  | 384       | 384               | 4.94                           | 13.43                          | 24.08                          |
| ViT_large_<br/>patch16_224 | 256       | 224               | 15.53                          | 49.50                          | 94.09                          |
| ViT_large_<br/>patch16_384 | 384       | 384               | 39.51                          | 152.46                         | 304.06                         |
| ViT_large_<br/>patch32_384 | 384       | 384               | 11.44                          | 36.09                          | 70.63                          |

| Models                               | Crop Size | Resize Short Size | FP32<br/>Batch Size=1<br/>(ms) | FP32<br/>Batch Size=4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
| ------------------------------------ | --------- | ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| DeiT_tiny_<br>patch16_224            | 256       | 224               | 3.61                           | 3.94                           | 6.10                           |
| DeiT_small_<br>patch16_224           | 256       | 224               | 3.61                           | 6.24                           | 10.49                          |
| DeiT_base_<br>patch16_224            | 256       | 224               | 6.13                           | 14.87                          | 28.50                          |
| DeiT_base_<br>patch16_384            | 384       | 384               | 14.12                          | 48.80                          | 97.60                          |
| DeiT_tiny_<br>distilled_patch16_224  | 256       | 224               | 3.51                           | 4.05                           | 6.03                           |
| DeiT_small_<br>distilled_patch16_224 | 256       | 224               | 3.70                           | 6.20                           | 10.53                          |
| DeiT_base_<br>distilled_patch16_224  | 256       | 224               | 6.17                           | 14.94                          | 28.58                          |
| DeiT_base_<br>distilled_patch16_384  | 384       | 384               | 14.12                          | 48.76                          | 97.09                          |

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/DeiT/` 中提供了模型的训练配置，可以通过如下脚本启动训练：此部分内容可以参考[ResNet50 模型训练、评估和预测](./ResNet.md#3)。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#4.1) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#4.2) 完成模型的推理预测。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。