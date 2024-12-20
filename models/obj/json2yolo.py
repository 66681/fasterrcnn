import json
import os
import numpy as np
from shapely.geometry import Polygon
import cv2
from torch.utils.data import Dataset, DataLoader
import torch

def convert_polygon_to_bbox(points):
    """
    将多边形点转换为边界框
    返回格式：[x_min, y_min, x_max, y_max]
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return [x_min, y_min, x_max, y_max]

def convert_bbox_to_yolo(bbox, image_width, image_height):
    """
    将边界框转换为YOLO格式 [x_center, y_center, width, height]
    所有值都归一化到[0, 1]范围
    """
    x_min, y_min, x_max, y_max = bbox
    
    # 计算中心点坐标和宽高
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # 归一化
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    
    return [x_center, y_center, width, height]

def convert_json_to_yolo(json_path, output_dir, class_mapping={'dseam': 0}):
    """
    转换单个json文件到YOLO格式
    
    Args:
        json_path: json文件路径
        output_dir: 输出目录
        class_mapping: 类别映射字典，默认将'dseam'映射为0
    """
    try:
        # 读取json文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        # 创建输出文件路径
        base_name = os.path.splitext(os.path.basename(data['imagePath']))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # 处理每个标注对象
        yolo_annotations = []
        for shape in data['shapes']:
            try:
                # 获取类别ID
                class_name = shape['label']
                if class_name not in class_mapping:
                    print(f"警告: 未知类别 {class_name} 在文件 {json_path} 中")
                    continue
                class_id = class_mapping[class_name]
                
                # 获取多边形点并转换为边界框
                points = shape['points']
                bbox = convert_polygon_to_bbox(points)
                
                # 转换为YOLO格式
                yolo_bbox = convert_bbox_to_yolo(bbox, image_width, image_height)
                
                # 添加类别ID和边界框信息
                yolo_line = f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}"
                yolo_annotations.append(yolo_line)
                
            except Exception as e:
                print(f"处理标注对象时出错: {str(e)}")
                continue
        
        # 写入YOLO格式文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_annotations))
        
        print(f"已转换: {json_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return False

def batch_convert(json_dir, output_dir, class_mapping={'dseam': 0}):
    """
    批量转换目录下的所有json文件
    
    Args:
        json_dir: json文件目录
        output_dir: 输出目录
        class_mapping: 类别映射字典
    """
    # 检查输入目录是否存在
    if not os.path.exists(json_dir):
        print(f"错误: 输入目录不存在: {json_dir}")
        print("请确保目录结构如下:")
        print("obj/")
        print("  ├── train_json/")
        print("  │   └── split/")
        print("  │       └── train/")
        print("  │           └── json/")
        print("  │               └── *.json")
        print("  └── train_yolo/")
        return
    
    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
    except Exception as e:
        print(f"创建输出目录时出错: {str(e)}")
        return
    
    # 获取所有json文件
    try:
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if not json_files:
            print(f"警告: 在 {json_dir} 中没有找到json文件")
            return
        total_files = len(json_files)
    except Exception as e:
        print(f"读取目录时出错: {str(e)}")
        return
    
    print(f"\n开始转换 {total_files} 个文件...")
    print(f"输入目录: {json_dir}")
    print(f"类别映射: {class_mapping}")
    
    # 转换计数
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # 处理每个文件
    for i, json_file in enumerate(json_files, 1):
        json_path = os.path.join(json_dir, json_file)
        print(f"\n处理文件 [{i}/{total_files}]: {json_file}")
        
        # 检查文件是否存在且不为空
        if not os.path.exists(json_path):
            print(f"跳过: 文件不存在 {json_file}")
            skipped_count += 1
            continue
            
        if os.path.getsize(json_path) == 0:
            print(f"跳过: 空文件 {json_file}")
            skipped_count += 1
            continue
        
        if convert_json_to_yolo(json_path, output_dir, class_mapping):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印详细的统计信息
    print("\n" + "="*50)
    print("转换完成!")
    print("="*50)
    print(f"成功转换: {success_count}")
    print(f"转换失败: {fail_count}")
    print(f"已跳过: {skipped_count}")
    print(f"总文件数: {total_files}")
    print(f"转换率: {(success_count/total_files*100):.1f}%")
    print(f"\n转换结果保存在: {output_dir}")
    print("="*50)

# 1. Dataset: 定义如何获取单个数据样本
class CustomDataset(Dataset):
    def __init__(self, data_list):
        """
        Dataset负责定义数据集的表示方式
        """
        self.data = data_list
    
    def __len__(self):
        """返回数据集的总大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """定义如何获取一个数据样本"""
        return self.data[idx]

# 2. 创建一个示例数据集
dataset = CustomDataset([i for i in range(100)])

# 3. DataLoader: 负责批处理，打乱，多进程加载等
dataloader = DataLoader(
    dataset=dataset,          # 使用我们定义的数据集
    batch_size=4,            # 每批处理的样本数
    shuffle=True,            # 是否打乱数据
    num_workers=2,           # 多进程加载
    drop_last=False          # 是否丢弃不完整的批次
)

# 4. 使用DataLoader迭代数据
for batch in dataloader:
    print(f"获取一个批次的数据: {batch}")

if __name__ == "__main__":
    try:
        # 设置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 训练集路径
        train_json_dir = os.path.join(current_dir, "split", "train", "json")
        train_output_dir = os.path.join(current_dir, "train_yolo")
        
        # 验证集路径
        val_json_dir = os.path.join(current_dir, "split", "val", "json")
        val_output_dir = os.path.join(current_dir, "val_yolo")
        
        print("\nJSON转YOLO格式转换工具")
        print("="*30)
        print(f"当前目录: {current_dir}")
        
        # 类别映射
        class_mapping = {
            'dseam': 0  # 将dseam类别映射为0
        }
        
        # 转换训练集
        print("\n处理训练集...")
        batch_convert(train_json_dir, train_output_dir, class_mapping)
        
        # 转换验证集
        print("\n处理验证集...")
        batch_convert(val_json_dir, val_output_dir, class_mapping)
        
    except KeyboardInterrupt:
        print("\n\n用户中断转换过程")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        raise 