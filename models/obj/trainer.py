import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ...models.config.config_tool import read_yaml
from ...models.obj.fasterrcnn_dataset import ObjectDetectionDataset
import os
from datetime import datetime
import gc
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def collate_fn(batch):
    """
    自定义收集函数，用于处理批次数据
    """
    return tuple(zip(*batch))

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0
    return iou

def compute_total_loss(loss_dict, device):
    """计算总损失"""
    total_loss = torch.tensor(0.0, device=device)
    
    # 累加所有损失
    for name, loss in loss_dict.items():
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=device)
        total_loss += loss
        
    return total_loss

def train_cfg(model, cfg):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        cfg: 配置信息，可以是配置文件路径或配置字典
    """
    # 如果cfg是路径，读取配置文件
    if isinstance(cfg, (str, bytes)):
        parameters = read_yaml(cfg)
        if parameters is None:
            raise ValueError(f"无法读取配置文件: {cfg}")
    else:
        parameters = cfg

    # 创建TensorBoard写入器
    log_dir = os.path.join(
        parameters['save']['log_dir'],
        datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard 日志目录: {log_dir}")
    print("\n" + "="*50)
    print("TensorBoard 服务器已自动启动")
    print("如果浏览器没有自动打开，请手动访问：")
    print("http://localhost:8088")
    print("\n如果访问出现问题，请手动运行：")
    print(f"tensorboard --logdir={log_dir} --port=8088")
    print("="*50 + "\n")

    # 创建数据加载器
    train_dataset = ObjectDetectionDataset(
        images_dir=parameters['dataset']['train_images_dir'],
        json_dir=parameters['dataset']['train_json_dir']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters['dataset']['batch_size'],
        shuffle=True,
        num_workers=parameters['dataset']['num_workers'],
        collate_fn=collate_fn,  # 使用命名函数而不是lambda
        pin_memory=True  # 启用内存固定以加速数据传输到GPU
    )
    
    # 设置优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parameters['training']['learning_rate'],
        momentum=parameters['training']['momentum'],
        weight_decay=parameters['training']['weight_decay']
    )
    
    # 设置学习率调度器
    if parameters['training'].get('lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=parameters['training']['lr_step_size'],
            gamma=parameters['training']['lr_gamma']
        )
    
    # 创建mAP计算器
    metric = MeanAveragePrecision()
    
    # 添加训练配置信息到TensorBoard
    writer.add_text('Training Config', f"""
    Batch Size: {parameters['dataset']['batch_size']}
    Learning Rate: {parameters['training']['learning_rate']}
    Momentum: {parameters['training']['momentum']}
    Weight Decay: {parameters['training']['weight_decay']}
    """, 0)
    
    # 训练循环
    global_step = 0
    total_batches = len(train_loader)
    
    # 获取总epoch数
    num_epochs = parameters['training']['epochs']
    
    # 添加epoch循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for images, targets in train_loader:
            try:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 将数据移到设备
                images = [image.to(model.device) for image in images]
                targets = [{k: v.to(model.device) for k, v in t.items()} 
                          for t in targets]
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                loss_dict = model(images, targets)

                # 检查损失类型
                for name, loss in loss_dict.items():
                    if not torch.is_tensor(loss):
                        print(f"警告: {name} 损失不是张量类型: {type(loss)}")
                        loss_dict[name] = torch.tensor(loss, device=model.device)

                # 计算总损失
                losses = compute_total_loss(loss_dict, model.device)


                # 反向传播
                try:
                    losses.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=1.0
                    )
                    # 更新参数
                    optimizer.step()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("GPU内存不足，尝试减小batch_size")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    raise e
                
                # 计算预测结果
                model.eval()
                with torch.no_grad():
                    predictions = model(images)
                model.train()
                
                # 更新mAP计算器
                metric.update(predictions, targets)
                
                # 计算平均IoU
                batch_ious = []
                for pred, target in zip(predictions, targets):
                    pred_boxes = pred['boxes']
                    target_boxes = target['boxes']
                    for pred_box in pred_boxes:
                        ious = [calculate_iou(pred_box.cpu(), target_box.cpu()) 
                               for target_box in target_boxes]
                        if ious:
                            batch_ious.append(max(ious))
                
                avg_iou = sum(batch_ious) / len(batch_ious) if batch_ious else 0
                
                # 直接记录原始值到TensorBoard
                writer.add_scalar('Loss/train', losses.item(), global_step)
                writer.add_scalar('Metrics/IoU', avg_iou, global_step)
                
                # 计算并记录mAP
                map_dict = metric.compute()
                writer.add_scalar('Metrics/mAP', map_dict['map'].item(), global_step)  # 平均mAP
                writer.add_scalar('Metrics/mAP_50', map_dict['map_50'].item(), global_step)  # IoU=0.5的mAP
                writer.add_scalar('Metrics/mAP_75', map_dict['map_75'].item(), global_step)  # IoU=0.75的mAP
                
                # 打印信息
                if global_step % 50 == 0:
                    print(f"\nEpoch {epoch+1}, Step {global_step}")
                    print(f"Loss: {losses.item():.4f}")
                    print(f"mAP: {map_dict['map'].item():.4f}")
                    print(f"mAP@0.5: {map_dict['map_50'].item():.4f}")
                    print(f"mAP@0.75: {map_dict['map_75'].item():.4f}")
                    print(f"IoU: {avg_iou:.4f}")
                else:
                    print(f"Epoch {epoch+1}, Step {global_step}/{total_batches*num_epochs}, "
                          f"Loss: {losses.item():.4f}", end='\r')
                
                global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("警告: GPU内存不足，尝试减小batch_size或图像大小")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise e
        
        # 每个epoch结束后的处理
        if parameters['training'].get('lr_scheduler', False):
            scheduler.step()
            
        # 记录每个epoch的学习率
        writer.add_scalar('Learning Rate', 
                         optimizer.param_groups[0]['lr'], 
                         global_step)
    
    # 关闭TensorBoard写入器
    writer.close()
