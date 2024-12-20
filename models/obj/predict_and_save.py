import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, save_image
import os
import numpy as np
from PIL import Image
from fasterrcnn import FasterRCNNModel
from fasterrcnn_dataset import ObjectDetectionDataset
from multivisionmodels.models.config.config_tool import read_yaml

def predict_and_save_images(model_path, config_path, save_dir):
    """
    加载训练好的模型并对图片进行预测和保存
    
    Args:
        model_path: 训练好的模型路径(.pth文件)
        config_path: 配置文件路径
        save_dir: 保存预测结果的目录
    """
    try:
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 读取配置文件
        cfg = read_yaml(config_path)
        if cfg is None:
            raise ValueError(f"无法读取配置文件: {config_path}")
        
        # 创建模型并加载权重
        model = FasterRCNNModel(num_classes=cfg['model']['num_classes'])
        model.load_weight(model_path)
        model.eval()  # 设置为评估模式
        
        # 创建数据集
        dataset = ObjectDetectionDataset(
            images_dir=cfg['dataset']['train_images_dir'],
            json_dir=cfg['dataset']['train_json_dir']
        )
        
        print(f"\n开始处理图片...")
        print(f"数据集大小: {len(dataset)} 个样本")
        
        # 处理每张图片
        for idx in range(len(dataset)):
            try:
                # 获取图片路径
                image_path = dataset.annotations[idx]['image_path']
                image_name = os.path.basename(image_path)
                
                # 读取图片
                image = read_image(image_path)
                
                # 确保图像是RGB格式
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                elif image.shape[0] == 4:
                    image = image[:3, :, :]
                
                # 进行预测
                with torch.no_grad():
                    # 将图像转换为浮点数并归一化到[0, 1]范围
                    img = image.float() / 255.0
                    
                    # 应用transforms
                    if model.transforms is not None:
                        img = model.transforms(img)
                    
                    # 移动到GPU
                    img = img.to(model.device)
                    
                    # 预测
                    predictions = model([img])
                
                # 获取预测结果
                pred = predictions[0]
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                
                # 筛选高置信度的预测
                mask = scores > 0.5
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]
                
                # 在图像上绘制边界框
                drawn_boxes = draw_bounding_boxes(
                    image,
                    boxes,
                    labels=[f"dseam {score:.2f}" for score in scores],
                    colors="red",
                    width=4,
                    font_size=30
                )
                
                # 保存结果
                save_path = os.path.join(save_dir, f"pred_{image_name}")
                save_image(drawn_boxes.float() / 255.0, save_path)
                
                # 打印进度
                print(f"处理进度: [{idx + 1}/{len(dataset)}] - {image_name}")
                print(f"检测到 {len(boxes)} 个目标，已保存到: {save_path}")
                
            except Exception as e:
                print(f"处理图片 {image_name} 时出错: {str(e)}")
                continue
        
        print("\n处理完成!")
        print(f"预测结果已保存到: {save_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "saved/best_model_loss_0.2378.pth")
    config_path = os.path.join(current_dir, "train.yaml")
    save_dir = os.path.join(current_dir, "train_pic")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 运行预测和保存
    predict_and_save_images(model_path, config_path, save_dir) 