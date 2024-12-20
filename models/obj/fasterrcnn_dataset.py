import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from PIL import Image
import os
import json
import numpy as np
from torchvision.io import read_image

class ObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, json_dir, transform=None):
        """
        自定义目标检测数据集，用于处理labelme格式的标注

        Args:
            images_dir (str): 图像文件夹路径
            json_dir (str): JSON标注文件夹路径
            transform (callable, optional): 可选的图像转换
        """
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.transform = transform
        self.annotations = []
        self.label_to_id = {"dseam": 1}  # 背景类为0
        
        # 获取所有JSON文件和图像文件
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        
        # 创建图像文件名到路径的映射
        image_map = {os.path.splitext(f)[0]: f for f in image_files}
        
        # 处理每个标注文件
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取对应的图像文件名
                json_base_name = os.path.splitext(json_file)[0]
                if json_base_name in image_map:
                    img_path = os.path.join(self.images_dir, image_map[json_base_name])
                else:
                    print(f"警告: 找不到对应的图像文件: {json_base_name}")
                    continue
                
                if not os.path.exists(img_path):
                    print(f"警告: 图像文件不存在: {img_path}")
                    continue
                
                # 处理标注
                boxes = []
                labels = []
                
                if 'shapes' not in data:
                    print(f"警告: JSON文件 {json_file} 中没有找到 'shapes' 字段")
                    continue
                
                for shape in data['shapes']:
                    if shape.get('shape_type') == 'polygon':
                        try:
                            # 从多边形点获取边界框
                            points = np.array(shape['points'])
                            x_min = float(points[:, 0].min())
                            y_min = float(points[:, 1].min())
                            x_max = float(points[:, 0].max())
                            y_max = float(points[:, 1].max())
                            
                            label = shape.get('label')
                            if label not in self.label_to_id:
                                print(f"警告: 未知标签 {label} 在文件 {json_file} 中")
                                continue
                            
                            # 添加边界框和标签
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(self.label_to_id[label])
                        except Exception as e:
                            print(f"处理标注时出错 {json_file}: {str(e)}")
                            continue
                
                if boxes:  # 只添加有标注的图像
                    self.annotations.append({
                        'image_path': img_path,
                        'boxes': boxes,
                        'labels': labels
                    })
            except Exception as e:
                print(f"处理JSON文件时出错 {json_file}: {str(e)}")
                continue
        
        print(f"成功加载 {len(self.annotations)} 个有效样本")
        if not self.annotations:
            raise ValueError("没有找到有效的训练数据！")

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # 获取图像和标注
        ann = self.annotations[idx]
        
        try:
            # 读取PNG图像
            image = read_image(ann['image_path'])
            
            # 确保图像是RGB格式
            if image.shape[0] == 1:  # 如果是灰度图
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:  # 如果是RGBA图
                image = image[:3, :, :]  # 只保留RGB通道
            
            # 将图像转换为浮点数并归一化到[0, 1]范围
            image = image.float() / 255.0
            
            # 转换为张量
            boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
            labels = torch.as_tensor(ann['labels'], dtype=torch.int64)
            
            # 创建目标字典
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
            
            # 应用transforms
            if self.transform is not None:
                image = self.transform(image)
            
            return image, target
            
        except Exception as e:
            print(f"处理图像时出错 {ann['image_path']}: {str(e)}")
            raise e

    def get_num_classes(self):
        """返回类别数量（包括背景）"""
        return len(self.label_to_id) + 1
