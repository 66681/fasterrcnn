import os

import PIL
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

from models.dataset_tool import masks_to_boxes, read_masks_from_txt, read_masks_from_pixels


class MaskRCNNDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, dataset_type=None, target_type='polygon'):
        self.data_path = dataset_path
        self.transforms = transforms
        self.img_path = os.path.join(dataset_path, "images/" + dataset_type)
        self.lbl_path = os.path.join(dataset_path, "labels/" + dataset_type)
        self.imgs = os.listdir(self.img_path)
        self.lbls = os.listdir(self.lbl_path)
        self.target_type = target_type
        self.deafult_transform= MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
        # print('maskrcnn inited!')

    def __getitem__(self, item):
        # print('__getitem__')
        img_path = os.path.join(self.img_path, self.imgs[item])
        lbl_path = os.path.join(self.lbl_path, self.imgs[item][:-3] + 'txt')
        img = PIL.Image.open(img_path).convert('RGB')
        # h, w = np.array(img).shape[:2]
        w, h = img.size
        # print(f'h,w:{h, w}')
        target = self.read_target(item=item, lbl_path=lbl_path, shape=(h, w))
        if self.transforms:
            img, target = self.transforms(img,target)
        else:
            img=self.deafult_transform(img)
        # print(f'img:{img.shape},target:{target}')
        return img, target

    def create_masks_from_polygons(self, polygons, image_shape):
        """创建一个与图像尺寸相同的掩码，并填充多边形轮廓"""
        colors = np.array([plt.cm.Spectral(each) for each in np.linspace(0, 1, len(polygons))])
        masks = []

        for polygon_data, col in zip(polygons, colors):
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            # 将多边形顶点转换为 NumPy 数组
            _, polygon = polygon_data
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

            # 使用 OpenCV 的 fillPoly 函数填充多边形
            # print(f'color:{col[:3]}')
            cv2.fillPoly(mask, [pts], np.array(col[:3]) * 255)
            mask = torch.from_numpy(mask)
            mask[mask != 0] = 1
            masks.append(mask)

        return masks

    def read_target(self, item, lbl_path, shape):
        # print(f'lbl_path:{lbl_path}')
        h, w = shape
        labels = []
        masks = []
        if self.target_type == 'polygon':
            labels, masks = read_masks_from_txt(lbl_path, shape)
        elif self.target_type == 'pixel':
            labels, masks = read_masks_from_pixels(lbl_path, shape)

        target = {}
        target["boxes"] = masks_to_boxes(torch.stack(masks))
        target["labels"] = torch.stack(labels)
        target["masks"] = torch.stack(masks)
        target["image_id"] = torch.tensor(item)
        target["area"] = torch.zeros(len(masks))
        target["iscrowd"] = torch.zeros(len(masks))
        return target

    def heatmap_enhance(self, img):
        # 直方图均衡化
        img_eq = cv2.equalizeHist(img)

        # 自适应直方图均衡化
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img_clahe = clahe.apply(img)

        # 将灰度图转换为热力图
        heatmap = cv2.applyColorMap(img_eq, cv2.COLORMAP_HOT)

    def __len__(self):
        return len(self.imgs)
