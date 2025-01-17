# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
from sklearn.metrics import confusion_matrix
import numpy as np
import paddle.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

def classification_eval(engine, epoch_id=0):
    if hasattr(engine.eval_metric_func, "reset"):
        engine.eval_metric_func.reset()
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.3f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".3f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system() == "Windows" else len(engine.eval_dataloader)
    
    gt_label = []
    pred_label = []
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0])
        
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")

        # image input
        if engine.amp and engine.amp_eval:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=engine.amp_level):
                out = engine.model(batch[0])
        else:
            out = engine.model(batch[0])

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        if isinstance(out, dict) and "Student" in out:
            out = out["Student"]
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]

        # gather Tensor when distributed
        if paddle.distributed.get_world_size() > 1:
            label_list = []
            device_id = paddle.distributed.ParallelEnv().device_id
            label = batch[1].cuda(device_id) if engine.config["Global"][
                "device"] == "gpu" else batch[1]
            paddle.distributed.all_gather(label_list, label)
            labels = paddle.concat(label_list, 0)

            if isinstance(out, list):
                preds = []
                for x in out:
                    pred_list = []
                    paddle.distributed.all_gather(pred_list, x)
                    pred_x = paddle.concat(pred_list, 0)
                    preds.append(pred_x)
            else:
                pred_list = []
                paddle.distributed.all_gather(pred_list, out)
                preds = paddle.concat(pred_list, 0)

            if accum_samples > total_samples and not engine.use_dali:
                if isinstance(preds, list):
                    preds = [
                        pred[:total_samples + current_samples - accum_samples]
                        for pred in preds
                    ]
                else:
                    preds = preds[:total_samples + current_samples -
                                  accum_samples]
                labels = labels[:total_samples + current_samples -
                                accum_samples]
                current_samples = total_samples + current_samples - accum_samples
        else:
            labels = batch[1:]
            preds  = out

        gt_label.append(batch[1].numpy())
        if len(out) == 2 and out[0].shape != out[1].shape:
            pred_label.append(out[0].numpy())
        else:
            pred_label.append(out.numpy())

        # calc loss
        if engine.eval_loss_func is not None:
            if engine.amp and engine.amp_eval:
                with paddle.amp.auto_cast(
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level=engine.amp_level):
                    loss_dict = engine.eval_loss_func(preds, labels)
            else:
                loss_dict = engine.eval_loss_func(preds, labels)

            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(float(loss_dict[key]), current_samples)

        #  calc metric
        if engine.eval_metric_func is not None:
            engine.eval_metric_func(preds, labels)
        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.3f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.3f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
                metric_msg = ""
            else:
                metric_msg = ", ".join([
                    "{}: {:.3f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()

    # 计算混淆矩阵
    pred_result = paddle.to_tensor(np.concatenate(pred_label))
    pred_score  = F.softmax(pred_result, axis=-1)
    pred_label  = pred_score.argsort(axis=1)[:,-1]
    gt_label    = np.concatenate(gt_label)
    cm          = confusion_matrix(gt_label, pred_label)
    print(cm)

    # 获取pred_label 对应的pred_score
    pred_score_max  = pred_score.max(axis=1)
    pred_score_mean = pred_score_max.mean()
    pred_score_max = np.array(pred_score_max)

    # 将混淆矩阵绘制成图像a，将pred_score_max的直方图绘制为图像b, 并对图像进行保存
    plt.figure(figsize=(12, 6), dpi=100)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.subplot(1, 2, 2)
    plt.hist(pred_score_max, bins=20, density=True)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution, mean={:.3f}'.format(np.mean(pred_score_max)))
    plt.grid(True)
    save_path = os.path.join(engine.config["Global"]["output_dir"], engine.config["Arch"]["name"], "eval")
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, "confusion_matrix_predict_score_{}.png".format(epoch_id))
    plt.savefig(save_name)

    if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
        metric_msg = ", ".join([
            "evalres: ma: {:.3f} label_f1: {:.3f} label_pos_recall: {:.3f} label_neg_recall: {:.3f} instance_f1: {:.3f} instance_acc: {:.3f} instance_prec: {:.3f} instance_recall: {:.3f}".
            format(*engine.eval_metric_func.attr_res())
        ])
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.attr_res()[0]
    else:
        metric_msg = ", ".join([
            "{}: {:.3f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.avg
