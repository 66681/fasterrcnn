import math
import os
import sys
from datetime import datetime
from typing import Mapping, Any
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes

from models.config.config_tool import read_yaml
from models.ins.trainer import train_cfg
from tools import utils


class MaskRCNNModel(nn.Module):

    def __init__(self, num_classes=0, transforms=None):
        super(MaskRCNNModel, self).__init__()
        self.__model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        if transforms is None:
            self.transforms = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
        if num_classes != 0:
            self.set_num_classes(num_classes)
            # self.__num_classes=0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, inputs):
        outputs = self.__model(inputs)
        return outputs

    def train(self, cfg):
        parameters = read_yaml(cfg)
        num_classes=parameters['num_classes']
        # print(f'num_classes:{num_classes}')
        self.set_num_classes(num_classes)
        train_cfg(self.__model, cfg)

    def set_num_classes(self, num_classes):
        in_features = self.__model.roi_heads.box_predictor.cls_score.in_features
        self.__model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
        in_features_mask = self.__model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.__model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
                                                                  num_classes=num_classes)

    def load_weight(self, pt_path):
        state_dict = torch.load(pt_path)
        self.__model.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.__model.load_state_dict(state_dict)
        # return super().load_state_dict(state_dict, strict)

    def predict(self, src, show_box=True, show_mask=True):
        self.__model.eval()

        img = read_image(src)
        img = self.transforms(img)
        img = img.to(self.device)
        result = self.__model([img])
        print(f'result:{result}')
        masks = result[0]['masks']
        boxes = result[0]['boxes']
        # cv2.imshow('mask',masks[0].cpu().detach().numpy())
        boxes = boxes.cpu().detach()
        drawn_boxes = draw_bounding_boxes((img * 255).to(torch.uint8), boxes, colors="red", width=5)
        print(f'drawn_boxes:{drawn_boxes.shape}')
        boxed_img = drawn_boxes.permute(1, 2, 0).numpy()
        # boxed_img=cv2.resize(boxed_img,(800,800))
        # cv2.imshow('boxes',boxed_img)

        mask = masks[0].cpu().detach().permute(1, 2, 0).numpy()

        mask = cv2.resize(mask, (800, 800))
        # cv2.imshow('mask',mask)
        img = img.cpu().detach().permute(1, 2, 0).numpy()

        masked_img = self.overlay_masks_on_image(boxed_img, masks)
        masked_img = cv2.resize(masked_img, (800, 800))
        cv2.imshow('img_masks', masked_img)
        # show_img_boxes_masks(img, boxes, masks)
        cv2.waitKey(0)

    def generate_colors(self, n):
        """
        生成n个均匀分布在HSV色彩空间中的颜色，并转换成BGR色彩空间。

        :param n: 需要的颜色数量
        :return: 一个包含n个颜色的列表，每个颜色为BGR格式的元组
        """
        hsv_colors = [(i / n * 180, 1 / 3 * 255, 2 / 3 * 255) for i in range(n)]
        bgr_colors = [tuple(map(int, cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0])) for hsv in hsv_colors]
        return bgr_colors

    def overlay_masks_on_image(self, image, masks, alpha=0.6):
        """
        在原图上叠加多个掩码，每个掩码使用不同的颜色。

        :param image: 原图 (NumPy 数组)
        :param masks: 掩码列表 (每个都是 NumPy 数组，二值图像)
        :param colors: 颜色列表 (每个颜色都是 (B, G, R) 格式的元组)
        :param alpha: 掩码的透明度 (0.0 到 1.0)
        :return: 叠加了多个掩码的图像
        """
        colors = self.generate_colors(len(masks))
        if len(masks) != len(colors):
            raise ValueError("The number of masks and colors must be the same.")

        # 复制原图，避免修改原始图像
        overlay = image.copy()

        for mask, color in zip(masks, colors):
            # 确保掩码是二值图像
            mask = mask.cpu().detach().permute(1, 2, 0).numpy()
            binary_mask = (mask > 0).astype(np.uint8) * 255  # 你可以根据实际情况调整阈值

            # 创建彩色掩码
            colored_mask = np.zeros_like(image)
            colored_mask[:] = color
            colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=binary_mask)

            # 将彩色掩码与当前的叠加图像混合
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

        return overlay


if __name__ == '__main__':
    # ins_model = MaskRCNNModel(num_classes=5)
    ins_model = MaskRCNNModel()
    # data_path = r'F:\DevTools\datasets\renyaun\1012\spilt'
    # ins_model.train(data_dir=data_path,epochs=5000,target_type='pixel',batch_size=6,num_workers=10,num_classes=5)
    ins_model.train(cfg='train.yaml')
