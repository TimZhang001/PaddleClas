import cv2
import json
import numpy as np
import os

# 将图片对应的分割标注json转化为二值化图像，并进行保存***.png -----------------------------
def get_mask_image_from_json(json_path):
   
    # 判断 JSON 文件是否存在
    if not os.path.exists(json_path):
        return None

    # 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # 获取图像高度和宽度
    img_height, img_width = json_data['imageHeight'], json_data['imageWidth'] 

    # 构造二值化图像
    mask_img = np.zeros((img_height, img_width), dtype=np.uint8)
    for shape in json_data['shapes']:
        points = shape['points']
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask_img, [points], color=255)  # 将形状区域填充为255

    mask_path = json_path.replace('.json', '_mask.png')
    
    cv2.imwrite(mask_path, mask_img)
    print(f'Mask image saved: {mask_path}')


# 产生空的标签 -----------------------------
def gen_empty_json_label(image_path):

    # 读取图像
    image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    base_name  = os.path.basename(image_path)

    # 图像宽度 高度 通道
    img_height, img_width = image_data.shape[0:2]

    # 构造json数据
    json_data = {
        "flags": {},
        "imageData": None,
        "imagePath": base_name,
        "imageHeight": img_height,
        "imageWidth": img_width,
        "shapes": [],
        "version": "SmartIBW_2.02.01",
    }

    # 构造json文件路径
    json_path = image_path.replace('.bmp', '.json').replace('.tif', '.json')
    json_str  = json.dumps(json_data, indent=4, separators=(',', ': '))
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)

    print(f'Empty label saved: {json_path}')


# 获取文件夹下所有的特定格式的文件  -----------------------------
def get_all_files_with_suffix(path, suffix_list=['.bmp']):
    items = os.listdir(path)
    file_list = []

    # 过滤出文件
    for item in items:
        cur_path = os.path.join(path, item)
        if not os.path.isfile(cur_path):
            continue

        for suffix in suffix_list:
            if cur_path.endswith(suffix):
                file_list.append(cur_path)
                break

    # 对文件进行排序
    file_list.sort()

    return file_list


# main函数
if __name__ == '__main__':
    root_paths  = "C:/Users/Administrator/Desktop/00_OK"  
    image_paths = get_all_files_with_suffix(root_paths, ['.tif', '.bmp'])

    for image_path in image_paths:
        gen_empty_json_label(image_path)