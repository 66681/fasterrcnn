import os
import shutil
import random
from pathlib import Path

def split_dataset(json_dir, image_dir, train_ratio=0.9, seed=42):
    """
    将数据集分成训练集和测试集
    
    参数:
        json_dir: JSON文件目录
        image_dir: 图片文件目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    # 设置随机种子确保可重复性
    random.seed(seed)
    
    # 创建目标文件夹
    output_dirs = {
        'train': {
            'json': Path('train/json'),
            'images': Path('train/images')
        },
        'val': {
            'json': Path('val/json'),
            'images': Path('val/images')
        }
    }
    
    # 创建所需的目录
    for split in output_dirs.values():
        for dir_path in split.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 随机打乱文件列表
    random.shuffle(json_files)
    
    # 计算训练集大小
    train_size = int(len(json_files) * train_ratio)
    
    # 分割数据集
    train_files = json_files[:train_size]
    test_files = json_files[train_size:]
    
    # 复制文件到对应目录
    def copy_files(file_list, split_type):
        for json_file in file_list:
            # 复制JSON文件
            src_json = Path(json_dir) / json_file
            dst_json = output_dirs[split_type]['json'] / json_file
            shutil.copy2(src_json, dst_json)
            
            # 获取对应的图片文件名（假设图片与JSON文件同名，仅扩展名不同）
            image_base = json_file.rsplit('.', 1)[0]
            # 支持多种图片格式
            for ext in ['.jpg', '.jpeg', '.png']:
                image_file = image_base + ext
                src_image = Path(image_dir) / image_file
                if src_image.exists():
                    dst_image = output_dirs[split_type]['images'] / image_file
                    shutil.copy2(src_image, dst_image)
                    break
    
    # 复制训练集和测试集文件
    copy_files(train_files, 'train')
    copy_files(test_files, 'val')
    
    # 打印数据集统计信息
    print(f"数据集分割完成！")
    print(f"训练集数量: {len(train_files)}")
    print(f"测试集数量: {len(test_files)}")

if __name__ == "__main__":
    # 设置源文件夹路径
    JSON_DIR = r"D:\python_ws\data\box_json"  # 请替换为实际的JSON文件夹路径
    IMAGE_DIR = r"D:\python_ws\data\pic"  # 请替换为实际的图片文件夹路径
    
    # 执行数据集分割
    split_dataset(JSON_DIR, IMAGE_DIR) 