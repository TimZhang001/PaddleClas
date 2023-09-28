import os
from PIL import Image
import numpy as np
import paddle
from paddle.io import Dataset
import cv2
import albumentations
import json
from scipy.ndimage.morphology import distance_transform_edt
import paddle.nn.functional as F

CLASS_NAMES_B11 = ['00_BgImage', '00_Crack', '01_Broken', '02_Bur', '03_Dirty'] # 'all'
CLASS_NAMES_LED = ['00_OK', '01_Mark', '02_Dianjiyichang', '03_Cuoceng', '04_Shuangbao']
CLASS_NAMES = CLASS_NAMES_LED # CLASS_NAMES_PRETRAINE

interpolationList = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}

def preprocess_mask(img):
    mask             = np.zeros_like(img)
    mask[img >= 1]   = 1
    
    # w * h -> w * h * c
    mask = mask.reshape((mask.shape[0], mask.shape[1], 1));

    return mask

class MVTecDataset(Dataset):
    def __init__(self, 
                 dataset_path='D:/dataset/mvtec_anomaly_detection', 
                 kind= 'train',
                 size=None, 
                 down_factor=4,
                 random_crop=False, 
                 interpolation="bicubic",
                 aug_flip        = False,
                 aug_rotate90    = False,
                 aug_scaleRotate = False,
                 aug_brightness_contrast = False,):
        
        self.dataset_path = dataset_path
        self.kind         = kind
        self.down_factor  = down_factor

        # load dataset -------------------------------------------------------------
        self.x, self.class_id, self.is_segment, self.mask = self.load_dataset_folder()
           
        # preprocess   -------------------------------------------------------------
        size      = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation  = interpolationList[interpolation]
            self.image_rescaler = albumentations.Resize( height=self.size, width = self.size)
            self.center_crop    = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)

        # auidmentation flip
        if aug_flip:
            self.aug_flip = albumentations.Compose([albumentations.VerticalFlip(p=0.5),              
                                                    albumentations.HorizontalFlip(p=0.5),])
        else:
            self.aug_flip = None

        if aug_rotate90:
            self.aug_rotate90 = albumentations.RandomRotate90(p=0.5)
        else:   
            self.aug_rotate90 = None

        # auidmentation ShiftScaleRotate
        if aug_scaleRotate:
            self.aug_scaleRotate = albumentations.ShiftScaleRotate(shift_limit=0.15, 
                                                                   scale_limit=0.15, 
                                                                   rotate_limit=45, p=0.5)
        else:
            self.aug_scaleRotate = None

        # auidmentation brightness_contrast
        if aug_brightness_contrast:
            self.aug_brightness_contrast = albumentations.RandomBrightnessContrast(brightness_limit=0.05, 
                                                                                   contrast_limit=0.05, p=0.5)
        else:
            self.aug_brightness_contrast = None

    def __getitem__(self, idx):
        xpath, class_id, is_segment, mask = self.x[idx], self.class_id[idx], self.is_segment[idx], self.mask[idx]
        
        if(class_id == 0 or not os.path.isfile(mask)):
            x,   mask   = self.sample_process_0(image=xpath)
            mask_weight = np.ones_like(mask)
        else:
            x, mask     = self.sample_process(image=xpath, mask=mask)
            mask_weight = self.distance_transform(mask, 1, 2)
            
            # 增加一个维度
            mask_weight = np.expand_dims(mask_weight, axis=0)
            mask        = mask.transpose((2, 0, 1))
    
        # downsize
        mask           = self.downsize(mask, self.down_factor)
        mask[mask > 0] = 1
        mask_weight    = self.downsize(mask_weight, self.down_factor)
        
        # w * h * c-> c * w * h 
        example                 = dict()
        example["image"]        = x.transpose((2, 0, 1))    
        example["segmentation"] = mask
        example["class_label"]  = paddle.to_tensor([class_id])
        example["is_segment"]   = paddle.to_tensor([is_segment])
        example["mask_weight"]  = mask_weight
        example["image_path"]   = xpath

        return example["image"], example["class_label"], example["is_segment"], example["segmentation"], example["mask_weight"], example["image_path"]

    def __len__(self):
        return len(self.x)
        
    def load_dataset_folder(self):
        
        x, class_id, is_segment, mask = [], [], [], []
        
        # 当前的路径
        imagepath = os.path.join(self.dataset_path, self.kind.lower())

        # 获取文件夹下所有的子文件夹
        subdirs = [x[0] for x in os.walk(imagepath)]

        # 针对每个子文件夹获取其中的所有图片文件（.bmp .tif .jpg）
        for subdir in subdirs:
            if subdir == imagepath:
                continue

            # 获取文件夹下的所有图片文件
            image_files = [f for f in os.listdir(subdir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif'))]

            # 获取subdir的文件夹名
            subdirname = os.path.basename(subdir)
            
            # 根据文件名判断文件的类别
            if subdirname in CLASS_NAMES :
                class_label =  CLASS_NAMES.index(subdirname)
                if class_label != 0:
                    class_label = 1
            else:
                assert False, "Unknown class name"
            
            # class_label 复制到和image_files一个维度
            class_label = [class_label] * len(image_files)
            
            class_id.extend(class_label)
            x.extend([os.path.join(subdir, f) for f in image_files])

        # 判断x中是否有json文件
        for i in range(len(x)):
            # 把x[i]的后缀更改为.json
            json_path = x[i].replace(os.path.splitext(x[i])[1], '.json')
            mask.append(json_path)
            
            # 判断json_path是否存在
            if(os.path.isfile(json_path)):
                is_segment.append(True)
            else:
                is_segment.append(False)
            

        return list(x), list(class_id), list(is_segment), list(mask)
        
    def sample_process_0(self, image):
        image = Image.open(image).convert('L')
        image = np.array(image).astype(np.uint8)

        # 如果维度不是3，就增加一个维度
        if len(image.shape) != 3:
            image = np.expand_dims(image, axis=2)
        
        # resize
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        # cropper    
        if self.size is not None:
            image = self.cropper(image=image)["image"]

        if self.size is not None:
            mask = np.zeros([self.size, self.size, 1], dtype=np.float32)
        else:
            mask = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.float32)
        
        # aug_flip
        if self.aug_flip is not None:
            image = self.aug_flip(image = image)["image"]

        # aug_rotate90
        if self.aug_rotate90 is not None:
            image = self.aug_rotate90(image = image)["image"]

        # aug_scaleRotate
        if self.aug_scaleRotate is not None:
            image = self.aug_scaleRotate(image = image)['image']
        
        # aug_brightness_contrast
        if self.aug_brightness_contrast is not None:
            image = self.aug_brightness_contrast(image = image)['image']
        
        # to [-1, 1]
        image = image.astype(np.float32)
        image = (image/255).astype(np.float32)
        
        return image, mask
    
    def sample_process(self, image, mask):
        image = Image.open(image).convert('L')
        image = np.array(image).astype(np.uint8)

        # 如果维度不是3，就增加一个维度
        if len(image.shape) != 3:
            image = np.expand_dims(image, axis=2)

        mask  = self.read_json_resize(mask, [self.size, self.size], False)
        mask  = np.array(mask).astype(np.float32)
   
        # resize
        if self.size is not None:
            processed = self.image_rescaler(image=image, mask=mask)
            image     = processed["image"]
            mask      = processed["mask"]

        # cropper    
        if self.size is not None:
            processed = self.cropper(image=image, mask=mask)
            image     = processed["image"]
            mask      = processed["mask"]
        
        # aug_flip
        if self.aug_flip is not None:
            processed = self.aug_flip(image = image, mask=mask)
            image     = processed["image"]
            mask      = processed["mask"]

        # aug_rotate90
        if self.aug_rotate90 is not None:
            processed = self.aug_rotate90(image = image, mask=mask)
            image     = processed["image"]
            mask      = processed["mask"]

        # aug_scaleRotate
        if self.aug_scaleRotate is not None:
            processed = self.aug_scaleRotate(image = image, mask=mask)
            image     = processed["image"]
            mask      = processed["mask"]
        
        # aug_brightness_contrast
        if self.aug_brightness_contrast is not None:
            processed = self.aug_brightness_contrast(image = image, mask=mask)
            image     = processed["image"]
            mask      = processed["mask"]

        # to [-1, 1]
        image = image.astype(np.float32)
        image = (image/255).astype(np.float32)
        mask  = preprocess_mask(mask)
        
        return image, mask
    
    def read_json_resize(self, path, resize_dim, dilate=None) -> (np.ndarray, bool):
        # 读取JSON文件
        with open(path, 'r') as f:
            data = json.load(f)

        # 获取图像尺寸
        image_width  = data['imageWidth']
        image_height = data['imageHeight']

        # 创建一个黑色背景图像
        image = np.ones((image_height, image_width), dtype=np.uint8) * 0

        # 获取形状信息并在图像上绘制
        shapes = data['shapes']
        for shape in shapes:
            points = shape['points']
            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(image, [points], color=255)  # 将形状区域填充为黑色

        if dilate is not None and dilate > 1:
            image = cv2.dilate(image, np.ones((dilate, dilate)))
        if resize_dim is not None:
            image = cv2.resize(image, dsize=resize_dim)
        
        # 二值化 大于0的设置为255
        image[image > 0] = 255
        return np.array((image / 255.0), dtype=np.float32)
    
    def distance_transform(self, mask: np.ndarray, max_val: float, p: float) -> np.ndarray:
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask[mask > 0] = 1
        h, w = mask.shape[:2]
        dst_trf = np.zeros((h, w))
        
        num_labels, labels = cv2.connectedComponents((mask * 255.0).astype(np.uint8), connectivity=8)
        for idx in range(1, num_labels):
            mask_roi= np.zeros((h, w))
            k = labels == idx
            mask_roi[k] = 255
            dst_trf_roi = distance_transform_edt(mask_roi)
            if dst_trf_roi.max() > 0:
                dst_trf_roi = (dst_trf_roi / dst_trf_roi.max())
                dst_trf_roi = (dst_trf_roi ** p) * max_val
            dst_trf += dst_trf_roi

        dst_trf[mask == 0] = 1
        dst_trf = np.array(dst_trf, dtype=np.float32)


        return dst_trf

    def downsize(self, image: np.ndarray, downsize_factor: int = 2) -> np.ndarray:
        img_t    = paddle.to_tensor(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
        
        # b h w c -> b c h w
        if len(img_t.shape)==4 and img_t.shape[3]==1:
            img_t    = img_t.transpose([0, 3, 1, 2])
        
        padding   = (downsize_factor, downsize_factor, downsize_factor, downsize_factor)
        pad_layer = paddle.nn.Pad2D(padding)

        img_t = pad_layer(img_t)

        kernel_size = 2 * downsize_factor + 1
        image_np    = F.avg_pool2d(img_t, kernel_size=kernel_size, stride=downsize_factor).numpy()
        return image_np[0] if len(image.shape) == 3 else image_np[0, 0]