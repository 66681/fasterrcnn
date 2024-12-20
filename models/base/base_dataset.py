from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from torchvision.transforms import  functional as F

class BaseDataset(Dataset, ABC):
    def __init__(self,dataset_path):
        self.default_transform=DefaultTransform()
        pass

    def __getitem__(self, index) -> T_co:
        pass

    @abstractmethod
    def read_target(self,item,lbl_path,extra=None):
        pass

    """显示数据集指定图片"""
    @abstractmethod
    def show(self,idx):
        pass

    """
    显示数据集指定名字的图片
    """

    @abstractmethod
    def show_img(self,img_path):
        pass

class DefaultTransform(nn.Module):
    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, torch.float)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[0.0, 1.0]``."
        )