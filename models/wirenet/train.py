def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard写入器
    """
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # 将数据移到设备
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        losses = model(images, targets)
        
        # 计算总损失
        loss = _loss(losses)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        
        # 打印训练信息
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: [{batch_idx}/{num_batches}], '
                  f'Loss: {loss.item():.4f}')
            
            # 记录到TensorBoard
            if writer is not None:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar('Loss/train', loss.item(), global_step)
                
                # 记录各个损失组件
                for loss_name, loss_value in losses.items():
                    if loss_name != "loss_wirepoint":
                        writer.add_scalar(f'Loss/{loss_name}', 
                                        loss_value.item(), 
                                        global_step)
                    else:
                        # 处理wirepoint损失
                        loss_labels = loss_value["losses"]
                        for label_loss in loss_labels[0].items():
                            name, value = label_loss
                            writer.add_scalar(f'Loss/wirepoint_{name}', 
                                            value.mean().item(), 
                                            global_step)
        
        # 清理内存
        del images, targets, losses, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 返回平均损失
    return total_loss / num_batches


def _loss(losses):
    """计算总损失
    
    Args:
        losses: 损失字典
    
    Returns:
        total_loss: 总损失
    """
    total_loss = 0
    
    # 处理非wirepoint损失
    for loss_name, loss_value in losses.items():
        if loss_name != "loss_wirepoint":
            total_loss += loss_value
    
    # 处理wirepoint损失
    if "loss_wirepoint" in losses:
        loss_labels = losses["loss_wirepoint"]["losses"]
        if loss_labels:  # 确保有损失值
            loss_labels_k = list(loss_labels[0].keys())
            for name in loss_labels_k:
                loss = loss_labels[0][name].mean()
                print(f"{name}: {loss.item():.4f}")
                total_loss += loss
    
    return total_loss

