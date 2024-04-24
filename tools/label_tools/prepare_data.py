# 将路径下的图片按照文件夹进行分类，生成对应的标签文件，用于训练
# 得到paddle训练需要的train_list.txt和val_list.txt label_list.txt

import os
import random
import matplotlib.pyplot as plt

# 获取路径下的所有文件夹，返回文件夹列表
def get_all_dir(path):
    all_dir = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            all_dir.append(dir)
    return all_dir

# 获取路径下的所有的图片文件，返回图片文件列表
def get_all_image(path):
    all_image = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.tif') or file.endswith('.bmp'):
                all_image.append(file)
    return all_image

# 生成标签文件
def generate_label_file(path, label_file):
    all_dir         = get_all_dir(path)
    label_file_path = os.path.join(path, label_file)
    
    with open(label_file_path, 'w') as f:
        for i, dir in enumerate(all_dir):
            f.write(dir + ' ' + str(i) + '\n')


    # 统计每个类别的图片数量，绘制直方图
    num_dict = {}
    for dir in all_dir:
        num_dict[dir] = len(get_all_image(os.path.join(path, dir)))
    print(num_dict)

    # 绘制直方图
    plt.bar(num_dict.keys(), num_dict.values())
    plt.xlabel('Class')
    plt.ylabel('Number')
    plt.title('Number of images in each class')
    plt.show()
    plt.savefig(os.path.join(path, 'class_num.png'))
    plt.close()


# 生成训练和验证的list文件
def generate_train_val_list(root_path, label_file, train_list, val_list, val_rate=0.1):
    
    # 获取类别的标签字典
    with open(os.path.join(root_path, label_file), 'r') as f:
        lines = f.readlines()
    label_dict = {}
    for line in lines:
        line = line.strip().split(' ')
        label_dict[line[0]] = line[1]
    
    
    # 获取所有的文件夹
    all_dir = get_all_dir(root_path)
    with open(os.path.join(root_path, train_list), 'w') as f1, open(os.path.join(root_path, val_list), 'w') as f2:
        for dir in all_dir:
            dir_path  = os.path.join(root_path, dir)
            all_image = get_all_image(dir_path)
            random.shuffle(all_image)
            for i, image in enumerate(all_image):
                if i < len(all_image) * val_rate:
                    f2.write(os.path.join(dir_path, image) + ' ' + label_dict[dir] + '\n')
                else:
                    f1.write(os.path.join(dir_path, image) + ' ' + label_dict[dir] + '\n')

if __name__ == '__main__':
    root_path  = '/raid/zhangss/dataset/Classify/MuraAD/'
    label_file = 'label_list.txt'
    train_list = 'train_list.txt'
    val_list   = 'val_list.txt'
    
    generate_label_file(root_path, label_file)
    generate_train_val_list(root_path, label_file, train_list, val_list, val_rate=0.15)
    print('Finish!')