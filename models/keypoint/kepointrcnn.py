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
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes

from models.config.config_tool import read_yaml
from models.keypoint.trainer import train_cfg

from tools import utils
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class KeypointRCNNModel(nn.Module):

    def __init__(self, num_classes=2,num_keypoints=2, transforms=None):
        super(KeypointRCNNModel, self).__init__()
        default_weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.__model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None,num_classes=num_classes,
                                                                              num_keypoints=num_keypoints,
                                                                              progress=False)
        if transforms is None:
            self.transforms = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
        # if num_classes != 0:
        #     self.set_num_classes(num_classes)
            # self.__num_classes=0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, inputs):
        outputs = self.__model(inputs)
        return outputs

    def train(self, cfg):
        parameters = read_yaml(cfg)
        num_classes = parameters['num_classes']
        num_keypoints = parameters['num_keypoints']
        # print(f'num_classes:{num_classes}')
        # self.set_num_classes(num_classes)
        self.num_keypoints = num_keypoints
        train_cfg(self.__model, cfg)

    # def set_num_classes(self, num_classes):
    #     in_features = self.__model.roi_heads.box_predictor.cls_score.in_features
    #     self.__model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    #
    #     # in_features_mask = self.__model.roi_heads.mask_predictor.conv5_mask.in_channels
    #     in_channels = self.__model.roi_heads.keypoint_predictor.
    #     hidden_layer = 256
    #     self.__model.roi_heads.mask_predictor = KeypointRCNNPredictor(in_channels, hidden_layer,
    #                                                               num_classes=num_classes)
    #     self.__model.roi_heads.keypoint_predictor=KeypointRCNNPredictor(in_channels, num_keypoints=num_classes)

    def load_weight(self, pt_path):
        state_dict = torch.load(pt_path)
        self.__model.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.__model.load_state_dict(state_dict)
        # return super().load_state_dict(state_dict, strict)


if __name__ == '__main__':
    # ins_model = MaskRCNNModel(num_classes=5)
    keypoint_model = KeypointRCNNModel(num_keypoints=2)
    # data_path = r'F:\DevTools\datasets\renyaun\1012\spilt'
    # ins_model.train(data_dir=data_path,epochs=5000,target_type='pixel',batch_size=6,num_workers=10,num_classes=5)
    keypoint_model.train(cfg='train.yaml')
