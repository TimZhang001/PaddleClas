import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import paddle.nn.functional as F
import paddle
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_cls_result(input_batch, output_batch, save_images=False, save_folder=None, dsize=[128, 128],):
    # ----- input -----
    image       = input_batch[0].detach().cpu().numpy()
    gt_label    = input_batch[1].detach().cpu().numpy()
    is_segment  = input_batch[2].detach().cpu().numpy()
    gt_mask     = input_batch[3].detach().cpu().numpy()
    mask_weight = input_batch[4].detach().cpu().numpy()
    image_path  = input_batch[5]
    
    # ----- input -----
    batch_size  = input_batch[0].shape[0]
            
    pred_clsprob = output_batch[0]
    pred_clsprob = F.softmax(pred_clsprob, axis=-1)
    pred_label   = pred_clsprob.argsort(axis=1)[:,-1]
    pred_clsprob = pred_clsprob.max(axis=1)

    pred_segprob = F.sigmoid(output_batch[1])
    pred_mask    = (pred_segprob > 0.5).astype('float32')  # 形状为 [1, 16, 16]

    # ----- to numpy -----
    pred_mask    = pred_mask.detach().cpu().numpy()
    pred_segprob = pred_segprob.detach().cpu().numpy()
    pred_label   = pred_label.detach().cpu().numpy()

    if save_images:
        for index in range(batch_size):                
            image_       = cv2.resize(np.transpose(image[index, :, :, :], (1, 2, 0)), dsize)
            mult_channel = False

            # 把image_的三个通道分开，然后在水平方向hconcat
            if(len(image_.shape)==3 and image_.shape[2] == 3):
                mult_channel = True

            xpath_       = image_path[index] 
            xpath_       = os.path.splitext(os.path.basename(xpath_))[0]  
            pred_seg_    = cv2.resize(pred_mask[index, 0, :, :], dsize)
            gt_mask_     = cv2.resize(gt_mask[index, 0, :, :], dsize)
            prob_seg_    = cv2.resize(pred_segprob[index, 0, :, :], dsize)
            mask_weight_ = cv2.resize(mask_weight[index, 0, :, :], dsize)
            
            plot_sample(image_, gt_label[index].item(), gt_mask_, 
                        pred_label[index].item(), pred_clsprob[index].item(),
                        pred_seg_,   prob_seg_, 
                        save_folder, mult_channel, img_name=xpath_)
        
    return pred_label, pred_clsprob, gt_label


def plot_sample(image, gt_label, gt_mask, 
                pred_label, cls_prob,
                pred_seg,   seg_prob, 
                save_dir,   mult_channel=False, img_name=None):
    
    # 只保留错误的，或者正确但是置信度比较低的
    if pred_label in range(0, 15):
        if gt_label != pred_label:
            if cls_prob > 0.25 and 1:      
                cur_save_dir = f"{save_dir}/wrong"
                os.makedirs(cur_save_dir, exist_ok=True)
            else:
                return
        else:
            if cls_prob < 0.5 and 1:
                cur_save_dir = f"{save_dir}/right"
                os.makedirs(cur_save_dir, exist_ok=True) 
            else:
                return
    else:
        return

    plt.figure()
    plt.clf()
    
    # ---------------input image -----------------------
    if mult_channel:
        total_num = 3 + 3
        for i in range(3):
            plt.subplot(1, total_num, i + 1)
            plt.xticks([]), plt.yticks([])
            plt.title('Input', fontsize=6)
            plt.imshow(image[:,:,i], cmap="gray")
    else:
        total_num = 3 + 1
        plt.subplot(1, total_num, 1)
        plt.xticks([]), plt.yticks([])
        plt.title('Input', fontsize=6)
        plt.imshow(image, cmap="gray")

    # ---------------GtMask -----------------------
    plt.subplot(1, total_num, total_num-2)
    plt.xticks([]), plt.yticks([])
    plt.title('GT-Mask', fontsize=6)
    plt.imshow(gt_mask, cmap="gray")

    plt.subplot(1, total_num, total_num-1)
    plt.xticks([]), plt.yticks([])
    plt.title(f"seg_prob: {seg_prob.max():.4f}", fontsize=6)
    plt.imshow(seg_prob, cmap="jet")

    plt.subplot(1, total_num, total_num)
    plt.xticks([]), plt.yticks([])
    plt.title('seg_rst', fontsize=6)
    plt.imshow((pred_seg * 255).astype(np.uint8), cmap="gray")
        
    out_prefix  = 'pred_{:02d}_'.format(pred_label) if pred_label is not None else out_prefix
    out_prefix  = out_prefix + '{:.3f}_'.format(cls_prob) if cls_prob is not None else ''
    rand_num    = '{:06d}'.format(np.random.randint(0, 100000))
    #image_label = image_label.data.cpu().numpy().squeeze()

    plt.savefig(f"{cur_save_dir}/{out_prefix}gt_{gt_label}_{img_name}_{rand_num}.jpg", bbox_inches='tight', dpi=150)
    plt.close()


def cal_confusion_matrix(pred_label_list, gt_label_list, pred_prob_list, save_folder =None, class_names=None):
    if isinstance(pred_label_list, list):
        pred_label_list = np.concatenate(pred_label_list)
    
    if isinstance(gt_label_list, list):
        gt_label_list = np.concatenate(gt_label_list)

    if isinstance(pred_prob_list, list):
        pred_prob_list = np.concatenate(pred_prob_list)

    if class_names is None:
        class_names = ['00_OK', '01_Shuangbao', '02_Modian', '03_Mark', '04_Tuoluo', '05_Kongdong', \
                      '06_Cuoceng-', '07_Ring-', '08_Zhenheng-', '09_Ganke-', '10_Dianjiyichang', '11_Huashang']

    # 生成混淆矩阵
    if len(gt_label_list.shape) == 2:
        gt_label_list = gt_label_list[:, 0]
    cm  = confusion_matrix(gt_label_list, pred_label_list)
    print(cm)

    # 计算每个类别的准确率
    acc = cm.diagonal()/cm.sum(axis=1)

    # 计算每个类别的召回率
    recall = cm.diagonal()/cm.sum(axis=0)

    # 计算每个类别的置信度平均值和标准差
    prob_mean = []
    prob_std  = []
    for i in range(len(class_names)):
        prob_mean.append(np.mean(pred_prob_list[gt_label_list==i]))
        prob_std.append(np.std(pred_prob_list[gt_label_list==i]))

    # 打印每个类别的信息
    for i in range(len(class_names)):
        print(f"{class_names[i]:<20}: acc={acc[i]:.4f}, recall={recall[i]:.4f}, prob_mean={prob_mean[i]:.4f}, prob_std={prob_std[i]:.4f}")

    # 计算整体的准确率
    acc_total = (gt_label_list==pred_label_list).sum()/len(gt_label_list)
    print(f"acc={acc_total:.4f}")

    # 计算整体的召回率
    recall_total = cm.diagonal().sum()/cm.sum()

    # 保存混淆矩阵
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.figure(figsize=(len(class_names), len(class_names)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(save_folder, "confusion_matrix.png"), dpi=200)
        plt.close()