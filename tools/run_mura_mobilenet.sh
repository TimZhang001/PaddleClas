job_name=MobileNetV3_large_x1_0 # 可修改，如 yolov7_tiny_300e_coco
project_name=Tim
config=ppcls/configs/${project_name}/${job_name}.yaml
log_dir=log_dir/${job_name}

# 1.训练（单卡/多卡），加 --eval 表示边训边评估，加 --amp 表示混合精度训练
CUDA_VISIBLE_DEVICES=5 python tools/train.py -c ${config}

# 2.评估，加 --classwise 表示输出每一类mAP
#CUDA_VISIBLE_DEVICES=5 python tools/eval.py -c ${config} 

# 3.预测 (单张图/图片文件夹）
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5

# 4.导出模型
#CUDA_VISIBLE_DEVICES=5 python tools/export_model.py -c ${config} 