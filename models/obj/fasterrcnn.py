import torch
from torch import nn
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Mapping, Any
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import subprocess
import time
import webbrowser
from threading import Thread
from torch.utils.data import DataLoader

from multivisionmodels.models.obj.trainer import train_cfg, collate_fn
from multivisionmodels.models.config.config_tool import read_yaml
from multivisionmodels.models.obj.fasterrcnn_dataset import  ObjectDetectionDataset


class FasterRCNNModel(nn.Module):
    def __init__(self, num_classes=0, transforms=None):
        super(FasterRCNNModel, self).__init__()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 初始化带预训练权重的FasterRCNN模型
        self.__model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        # 设置默认transforms、对图像数据集进行预处理(Resize等)
        if transforms is None:
            self.transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
        else:
            self.transforms = transforms
        
        # 如果指定了类别数，更新分类器
        if num_classes != 0:
            self.set_num_classes(num_classes)

        # 将模型移到GPU
        self.__model.to(self.device)

    def forward(self, images, targets=None):
        """
        前向传播
        
        Args:
            images: 输入图像
            targets: 训练时的目标数据，推理时为None
            
        Returns:
            训练模式：返回损失字典，每个损失都是张量
            推理模式：返回预测结果
        """
        if self.training and targets is not None:
            loss_dict = self.__model(images, targets)
            # 确保所有损失都是张量
            for name, loss in loss_dict.items():
                if not torch.is_tensor(loss):
                    loss_dict[name] = torch.tensor(loss, device=self.device)
            return loss_dict
        else:
            return self.__model(images)

    def train(self, mode=True):
        """
        重写train方法以支持两种调用方式
        """
        if isinstance(mode, bool):
            # 调用父类的train方法
            super().train(mode)
            self.__model.train(mode)
            return self
        else:
            # 如果传入的是配置，则进行训练
            cfg = mode
            if isinstance(cfg, str):
                parameters = read_yaml(cfg)
                if parameters is None:
                    raise ValueError(f"无法读取配置文件: {cfg}")
            else:
                parameters = cfg

            if 'model' not in parameters or 'num_classes' not in parameters['model']:
                raise ValueError("配置中少 'model.num_classes' 参数")

            num_classes = parameters['model']['num_classes']
            self.set_num_classes(num_classes)
            train_cfg(self, parameters)

    def set_num_classes(self, num_classes):
        """设置类别数"""
        in_features = self.__model.roi_heads.box_predictor.cls_score.in_features
        self.__model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def load_weight(self, pt_path):
        """
        加载模型权重
        
        Args:
            pt_path: 权��文件路径
        """
        try:
            # 加载权重文件
            checkpoint = torch.load(pt_path)
            
            # 如果是完整的检查点（包含额外信息），则提取模型状态字典
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"\n加载检查点信息:")
                print(f"轮次: {checkpoint.get('epoch', 'unknown')}")
                print(f"测试损失: {checkpoint.get('test_loss', 'unknown'):.4f}")
                print(f"时间戳: {checkpoint.get('timestamp', 'unknown')}")
            else:
                state_dict = checkpoint
            
            # 处理状态字典的键名
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_FasterRCNNModel__model.'):
                    # 移除前缀
                    new_key = key.replace('_FasterRCNNModel__model.', '')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # 加载处理后的权重
            self.__model.load_state_dict(new_state_dict)
            print(f"\n成功加载权重: {pt_path}")
            
        except Exception as e:
            print(f"\n加载权重文件时出错: {str(e)}")
            print("尝试使用非严格模式加载...")
            try:
                self.__model.load_state_dict(new_state_dict, strict=False)
                print("使用非严格模式成功加载权重")
            except Exception as e2:
                print(f"非严格模式加载也失败: {str(e2)}")
                raise

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """加载状态字典"""
        # 处理键名前缀问题
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_FasterRCNNModel__model.'):
                # 移除前缀
                new_key = key.replace('_FasterRCNNModel__model.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
            
        # 加载处理后的状态字典
        self.__model.load_state_dict(new_state_dict)

    def predict(self, src, show_box=True):
        """预测并可视化"""
        self.eval()  # 设置为评估模式
        
        # 读取并转换图像
        img = read_image(src)
        
        # 确保图像是RGB格式
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3, :, :]
        
        # 将图像转换为浮点数并归一化到[0, 1]范围
        img = img.float() / 255.0
        
        # 应用transforms
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = img.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            result = self([img])
        
        # 获取预测框
        boxes = result[0]['boxes']
        scores = result[0]['scores']
        labels = result[0]['labels']
        
        # 可视化
        if show_box:
            boxes = boxes.cpu().detach()
            # 将图像转回uint8以进行可视化
            vis_img = (img * 255).to(torch.uint8)
            drawn_boxes = draw_bounding_boxes(
                vis_img, 
                boxes, 
                colors="red",
                width=5
            )
            # 使用matplotlib显示
            plt.figure(figsize=(12, 8))
            plt.imshow(drawn_boxes.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.show()
        
        return boxes, scores, labels

def start_tensorboard(logdir, port=8088):
    """
    启动TensorBoard服务器
    """
    try:
        # 启动TensorBoard进程
        subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)])
        print(f"\nTensorBoard 服务器正在启动，端口: {port}")
        print("等待服务器启动...")

        
        # # 打开浏览器
        # url = f"http://localhost:{port}"
        # webbrowser.open(url)
        # print(f"已自动打开浏览器访问 {url}")
    except Exception as e:
        print(f"启动TensorBoard时出错: {str(e)}")
        print("请手动运行以下命令：")
        print(f"tensorboard --logdir={logdir} --port={port}")

if __name__ == '__main__':
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'train.yaml')
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 读取配置文件
        cfg = read_yaml(config_path)
        if cfg is None:
            raise ValueError(f"无法读取配置文件: {config_path}")
            
        print("配置文件路径:", config_path)
        print("配置文件内容:", cfg)

        # 1. 创建训练数据集
        train_dataset = ObjectDetectionDataset(
            images_dir=cfg['dataset']['train_images_dir'],
            json_dir=cfg['dataset']['train_json_dir']
        )

        # 创建测试数据集
        test_dataset = ObjectDetectionDataset(
            images_dir=cfg['dataset']['test_images_dir'],
            json_dir=cfg['dataset']['test_json_dir']
        )
        
        # 2. 创建模型
        model = FasterRCNNModel(num_classes=cfg['model']['num_classes'])
        
        # 加载预训练模型（如果配置了的话）
        start_epoch = 0
        best_test_loss = float('inf')
        if cfg.get('pretrain', {}).get('use_pretrain', False):
            checkpoint_path = cfg['pretrain']['checkpoint_path']
            if os.path.exists(checkpoint_path):
                print(f"\n加载预训练模型: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_test_loss = checkpoint.get('test_loss', float('inf'))
                print(f"从 epoch {start_epoch} 继续训练")
                print(f"之前的最佳测试损失: {best_test_loss:.4f}")
            else:
                print(f"\n警告: 预训练模型文件不存在: {checkpoint_path}")
                print("将从头开始训练")
        
        # 3. 训练和测试循环
        save_dir = cfg['save']['model_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建TensorBoard写入器
        log_dir = os.path.join(
            cfg['save']['log_dir'],
            datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")
        
        # 在新线程中启动TensorBoard
        tensorboard_thread = Thread(
            target=start_tensorboard,
            args=(cfg['save']['log_dir'], 8088)
        )
        tensorboard_thread.daemon = True  # 设置为守护线程，这样主程序退出时会自动结束
        tensorboard_thread.start()
        
        for epoch in range(start_epoch, cfg['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")



            # 训练阶段
            model.train(True)
            train_cfg(model, cfg)

            # 测试阶段
            model.train(False)  # 设置为评估模式
            test_loss = 0.0
            total_tests = 0
            
            print("\n" + "="*50)
            print("开始测试阶段...")
            print("="*50)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg['dataset']['batch_size'],
                shuffle=False,
                num_workers=cfg['dataset']['num_workers'],
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            print(f"测试数据集大小: {len(test_dataset)} 个样本")
            print(f"批次大小: {cfg['dataset']['batch_size']}")
            print(f"总批次数: {len(test_loader)}\n")
            
            # 进行测试
            with torch.no_grad():
                for batch_idx, (test_images, test_targets) in enumerate(test_loader, 1):
                    try:
                        # 将数据移到GPU
                        test_images = [image.to(model.device) for image in test_images]
                        test_targets = [{k: v.to(model.device) for k, v in t.items()} for t in test_targets]
                        
                        # 临时将模型设置为训练模式以获取损失
                        model.train()
                        loss_dict = model(test_images, test_targets)
                        print(loss_dict)
                        model.train(False)  # 恢复为评估模式
                        
                        # 计算总损失
                        batch_loss = sum(loss.item() for loss in loss_dict.values())
                        test_loss += batch_loss
                        total_tests += 1
                        
                        # 打印详细的损失信息
                        if batch_idx % 5 == 0:
                            print("-"*30)
                            print(f"批次: [{batch_idx}/{len(test_loader)}]")
                            print(f"当前批次损失: {batch_loss:.4f}")
                            # 打印各个损失组件
                            for loss_name, loss_value in loss_dict.items():
                                print(f"{loss_name}: {loss_value.item():.4f}")
                            
                            if torch.cuda.is_available():
                                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                                print(f"GPU 内存使用: {memory_allocated:.2f}GB (已分配) / {memory_reserved:.2f}GB (已预留)")
                            print("-"*30 + "\n")
                            
                    except Exception as e:
                        print(f"\n错误: 处理测试批次 {batch_idx} 时出错:")
                        print(f"错误类型: {type(e).__name__}")
                        print(f"错误信息: {str(e)}")
                        print("跳过此批次...\n")
                        continue

            # 计算平均测试损失
            avg_test_loss = test_loss / total_tests if total_tests > 0 else float('inf')
            
            print("\n" + "="*50)
            print("测试阶段完成！")
            print("="*50)
            print(f"成功处理的批次数: {total_tests}/{len(test_loader)}")
            print(f"平均测试损失: {avg_test_loss:.4f}")
            if torch.cuda.is_available():
                print(f"最终 GPU 内存使用: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
            print("="*50 + "\n")
            
            # 记录测试损失到TensorBoard
            writer.add_scalar('Loss/val', avg_test_loss, epoch)
            
            # 获取当前时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 只在损失值更好时保存模型
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_path = os.path.join(
                    save_dir, 
                    f'best_model_loss_{avg_test_loss:.4f}.pth'
                )
                
                # 保存模型状态
                model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_loss': avg_test_loss,
                    'timestamp': timestamp
                }
                
                # 保存模型
                torch.save(model_state, best_model_path)
                print(f"\n发现更好的模型！")
                print(f"当前最佳损失: {best_test_loss:.4f}")
                print(f"保存模型到: {best_model_path}")
        
        # 关闭TensorBoard写入器
        writer.close()
        
        print("\n训练完成!")
        print(f"最佳测试损失: {best_test_loss:.4f}")

        # 4. 使用最佳模型进行预测演示
        # 加载最佳模型
        best_model_path = os.path.join(save_dir, 'best_fasterrcnn_model.pth')
        model.load_weight(best_model_path)
        
        # 从测试集随机选择一张图片进行预测
        test_idx = np.random.randint(len(test_dataset))
        test_image_path = test_dataset.annotations[test_idx]['image_path']
        
        # 进行预测
        boxes, scores, labels = model.predict(test_image_path)
        
        # 打印预测结果
        print(f"\n预测图片: {test_image_path}")
        print("预测结果：")
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:  # 置信度阈值
                print(f"类别: {label}, 置信度: {score:.2f}, 边界框: {box}")

    except Exception as e:
        print(f"错误: {str(e)}")
        raise