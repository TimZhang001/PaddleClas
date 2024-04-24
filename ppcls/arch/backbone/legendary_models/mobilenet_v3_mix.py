# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://arxiv.org/abs/1905.02244

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn import functional as F

from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "MobileNetV3_small_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_pretrained.pdparams",
    "MobileNetV3_small_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_5_pretrained.pdparams",
    "MobileNetV3_small_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_75_pretrained.pdparams",
    "MobileNetV3_small_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_pretrained.pdparams",
    "MobileNetV3_small_x1_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_25_pretrained.pdparams",
    "MobileNetV3_large_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_35_pretrained.pdparams",
    "MobileNetV3_large_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_5_pretrained.pdparams",
    "MobileNetV3_large_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_75_pretrained.pdparams",
    "MobileNetV3_large_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_pretrained.pdparams",
    "MobileNetV3_large_x1_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_25_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {
    "MobileNetV3_small":
    ["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    "MobileNetV3_large":
    ["blocks[0]", "blocks[2]", "blocks[5]", "blocks[11]", "blocks[14]"]
}

__all__ = MODEL_URLS.keys()

# "large", "small" is just for MobinetV3_large, MobileNetV3_small respectively.
# The type of "large" or "small" config is a list. Each element(list) represents a depthwise block, which is composed of k, exp, se, act, s.
# k: kernel_size
# exp: middle channel number in depthwise block
# c: output channel number in depthwise block
# se: whether to use SE block
# act: which activation to use
# s: stride in depthwise block
NET_CONFIG = {
    "large": [
        #idx k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],       #
        [3, 64, 24, False, "relu", 1],       # 1/2   2->1
        [3, 72, 24, False, "relu", 1],       #       return 0
        [5, 72, 40, True, "relu", 2],        # 1/4   
        [5, 120, 40, True, "relu", 1],       #
        [5, 120, 40, True, "relu", 1],       #       return 1
        [3, 240, 80, False, "hardswish", 2], # 1/8   
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1], 
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1], #       return 2
        [5, 672, 160, True, "hardswish", 2], # 1/16  
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1], #       return 3
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 1],        # 1/2   return 0 2->1
        [3, 72, 24, False, "relu", 2],       # 1/4   
        [3, 88, 24, False, "relu", 1],       #       return 1
        [5, 96, 40, True, "hardswish", 2],   # 1/8   
        [5, 240, 40, True, "hardswish", 1],  #
        [5, 240, 40, True, "hardswish", 1],  # 
        [5, 120, 48, True, "hardswish", 1],  # 
        [5, 144, 48, True, "hardswish", 1],  #       return 2
        [5, 288, 96, True, "hardswish", 2],  # 1/16  2->1
        [5, 576, 96, True, "hardswish", 1],  #
        [5, 576, 96, True, "hardswish", 1],  #       return 3
    ]
}
# first conv output channel number in MobileNetV3
STEM_CONV_NUMBER = 16
# last second conv output channel for "small"
LAST_SECOND_CONV_SMALL = 576
# last second conv output channel for "large"
LAST_SECOND_CONV_LARGE = 960
# last conv output channel number for "large" and "small"
LAST_CONV = 1280

# Segment OUT_INDEX
OUT_INDEX = {"large": [2, 5, 11, 14], "small": [0, 2, 7, 10]}

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class MobileNetV3_Mix(TheseusLayer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
        class_num: int=1000. The number of classes.
        inplanes: int=16. The output channel number of first convolution layer.
        class_squeeze: int=960. The output channel number of penultimate convolution layer. 
        class_expand: int=1280. The output channel number of last convolution layer. 
        dropout_prob: float=0.2.  Probability of setting units to zero.
    Returns:
        model: nn.Layer. Specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 config,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 input_channel=3,
                 inplanes=STEM_CONV_NUMBER,
                 class_squeeze=LAST_SECOND_CONV_LARGE,
                 class_expand=LAST_CONV,
                 dropout_prob=0.2,
                 out_index=OUT_INDEX["small"],
                 return_patterns=None,
                 return_stages=None,
                 export_model=False,
                 is_train=True,
                 **kwargs):
        super().__init__()

        self.cfg           = config
        self.scale         = scale
        self.inplanes      = inplanes
        self.class_squeeze = class_squeeze
        self.class_expand  = class_expand
        self.class_num     = class_num
        self.out_index     = out_index
        self.input_channel = input_channel

        # --------------- feature extraction ---------------
        self.conv = ConvBNLayer(
            in_c=self.input_channel,
            out_c=_make_divisible(self.inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.blocks = nn.Sequential(* [
            ResidualUnit(
                in_c=_make_divisible(self.inplanes * self.scale if i == 0 else
                                     self.cfg[i - 1][2] * self.scale),
                mid_c=_make_divisible(self.scale * exp),
                out_c=_make_divisible(self.scale * c),
                filter_size=k,
                stride=s,
                use_se=se,
                act=act) for i, (k, exp, c, se, act, s) in enumerate(self.cfg)
        ])

        # --------------- segment ---------------
        out_channels = [config[idx][2] for idx in self.out_index]
        self.feat_channels = [
            _make_divisible(self.scale * c) for c in out_channels
        ]
        self.seg_header = MoblieSegmentHead(self.feat_channels, 1)
        
        # --------------- classifier ---------------
        self.last_second_conv = ConvBNLayer(
            in_c=_make_divisible(self.cfg[-1][2] * self.scale),
            out_c=_make_divisible(self.scale * self.class_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.avg_pool = AdaptiveAvgPool2D(1)
        self.last_conv = Conv2D(
            in_channels=_make_divisible(self.scale * self.class_squeeze),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)

        self.hardswish = nn.Hardswish()
        if dropout_prob is not None:
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.dropout = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc      = Linear(self.class_expand + 2, class_num)

        self.is_train = is_train
        self.normalize_in_model = export_model
        if self.normalize_in_model:
            # BCHW 1x3x1x1
            mean = [0, 0, 0]
            std  = [1, 1, 1]
            mean = paddle.to_tensor(mean)
            std  = paddle.to_tensor(std)
            mean = paddle.unsqueeze(mean, axis=1)
            mean = paddle.unsqueeze(mean, axis=2)
            mean = paddle.unsqueeze(mean, axis=0)

            std = paddle.unsqueeze(std, axis=1)
            std = paddle.unsqueeze(std, axis=2)
            std = paddle.unsqueeze(std, axis=0)

            self.mean = mean
            self.std  = std

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        if self.normalize_in_model:
            x = (x / 255.0 - self.mean) / self.std
        else:
            x = x
        
        x = self.conv(x)
        feat_list = []

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                feat_list.append(x)

        # ---- seg header -----
        seg = self.seg_header(feat_list)

        # 1 + 1
        global_max_seg = paddle.max(paddle.max(seg, axis=-1, keepdim=True), axis=-2, keepdim=True)
        global_avg_seg = paddle.mean(seg, axis=(-1, -2), keepdim=True)

        # ---- cls header -----
        x = self.last_second_conv(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = paddle.concat([global_max_seg, global_avg_seg, x], axis=1)
        x = self.flatten(x)
        x = self.fc(x)

        if self.is_train:
            return x, seg
        else:
            return x


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None):
        super().__init__()

        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(TheseusLayer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


# nn.Hardsigmoid can't transfer "slope" and "offset" in nn.functional.hardsigmoid
class Hardsigmoid(TheseusLayer):
    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return nn.functional.hardsigmoid(
            x, slope=self.slope, offset=self.offset)


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)

class MoblieSegmentHead(TheseusLayer):
    def __init__(self, backbone_channels, num_outputs):
        super(MoblieSegmentHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2D(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2D(last_inp_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
    
    def forward(self, x):
        x0_h, x0_w = x[0].shape[2], x[0].shape[3]
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = paddle.concat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x 

class SegmentationHead(TheseusLayer):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.num_classes = num_classes
        self.conv1 = ConvBNLayer(in_channels, in_channels, filter_size=5, stride=1, padding=2, if_act=True, act="relu")
        self.conv2 = ConvBNLayer(in_channels, num_classes, filter_size=3, stride=1, padding=1, if_act=False, act=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def MobileNetV3_Mix_small_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_35` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["small"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        out_index=OUT_INDEX["small"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x0_35"],
                     use_ssld)
    return model


def MobileNetV3_Mix_small_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_5` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["small"],
        scale=0.5,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        out_index=OUT_INDEX["small"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x0_5"],
                     use_ssld)
    return model


def MobileNetV3_Mix_small_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_75
    Args:
        pretrained: bool=false or str. if `true` load pretrained parameters, `false` otherwise.
                    if str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_75` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["small"],
        scale=0.75,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        out_index=OUT_INDEX["small"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x0_75"],
                     use_ssld)
    return model


def MobileNetV3_Mix_small_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x1_0` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["small"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        out_index=OUT_INDEX["small"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x1_0"],
                     use_ssld)
    return model


def MobileNetV3_Mix_small_x1_25(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x1_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x1_25` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["small"],
        scale=1.25,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        out_index=OUT_INDEX["small"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x1_25"],
                     use_ssld)
    return model


def MobileNetV3_Mix_large_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x0_35` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["large"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=OUT_INDEX["large"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x0_35"],
                     use_ssld)
    return model


def MobileNetV3_Mix_large_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x0_5` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["large"],
        scale=0.5,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=OUT_INDEX["large"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x0_5"],
                     use_ssld)
    return model


def MobileNetV3_Mix_large_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x0_75` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["large"],
        scale=0.75,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=OUT_INDEX["large"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x0_75"],
                     use_ssld)
    return model


def MobileNetV3_Mix_large_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x1_0` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["large"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=OUT_INDEX["large"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x1_0"],
                     use_ssld)
    return model


def MobileNetV3_Mix_large_x1_25(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x1_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x1_25` model depends on args.
    """
    model = MobileNetV3_Mix(
        config=NET_CONFIG["large"],
        scale=1.25,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=OUT_INDEX["large"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x1_25"],
                     use_ssld)
    return model
