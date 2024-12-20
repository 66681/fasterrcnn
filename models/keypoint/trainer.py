import math
import os
import sys
from datetime import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

from models.config.config_tool import read_yaml
from models.ins.maskrcnn_dataset import MaskRCNNDataset
from models.keypoint.keypoint_dataset import KeypointDataset
from tools import utils, presets
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print(f'images:{images}')
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            print(f'loss_dict:{loss_dict}')
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def train_cfg(model, cfg):
    parameters = read_yaml(cfg)
    print(f'train parameters:{parameters}')
    train(model, **parameters)

def train(model, **kwargs):
    # 默认参数
    default_params = {
        'dataset_path': '/path/to/dataset',
        'num_classes': 2,
        'num_keypoints':2,
        'opt': 'adamw',
        'batch_size': 2,
        'epochs': 10,
        'lr': 0.005,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'lr_step_size': 3,
        'lr_gamma': 0.1,
        'num_workers': 4,
        'print_freq': 10,
        'target_type': 'polygon',
        'enable_logs': True,
        'augmentation': False,
        'checkpoint':None
    }
    # 更新默认参数
    for key, value in kwargs.items():
        if key in default_params:
            default_params[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    # 解析参数
    dataset_path = default_params['dataset_path']
    num_classes = default_params['num_classes']
    batch_size = default_params['batch_size']
    epochs = default_params['epochs']
    lr = default_params['lr']
    momentum = default_params['momentum']
    weight_decay = default_params['weight_decay']
    lr_step_size = default_params['lr_step_size']
    lr_gamma = default_params['lr_gamma']
    num_workers = default_params['num_workers']
    print_freq = default_params['print_freq']
    target_type = default_params['target_type']
    augmentation = default_params['augmentation']
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_result_ptath = os.path.join('train_results', datetime.now().strftime("%Y%m%d_%H%M%S"))
    wts_path = os.path.join(train_result_ptath, 'weights')
    tb_path = os.path.join(train_result_ptath, 'logs')
    writer = SummaryWriter(tb_path)

    transforms = None
    # default_transforms = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    if augmentation:
        transforms = get_transform(is_train=True)
        print(f'transforms:{transforms}')
    if not os.path.exists('train_results'):
        os.mkdir('train_results')

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = KeypointDataset(dataset_path=dataset_path,
                              transforms=transforms, dataset_type='train', target_type=target_type)
    dataset_test = KeypointDataset(dataset_path=dataset_path, transforms=None,
                                   dataset_type='val')

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    train_collate_fn = utils.collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=num_workers, collate_fn=train_collate_fn
    )
    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, sampler=test_sampler, num_workers=num_workers, collate_fn=utils.collate_fn
    # )

    img_results_path = os.path.join(train_result_ptath, 'img_results')
    if os.path.exists(train_result_ptath):
        pass
    #     os.remove(train_result_ptath)
    else:
        os.mkdir(train_result_ptath)

    if os.path.exists(train_result_ptath):
        os.mkdir(wts_path)
        os.mkdir(img_results_path)

    for epoch in range(epochs):
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, None)
        losses = metric_logger.meters['loss'].global_avg
        print(f'epoch {epoch}:loss:{losses}')
        if os.path.exists(f'{wts_path}/last.pt'):
            os.remove(f'{wts_path}/last.pt')
        torch.save(model.state_dict(), f'{wts_path}/last.pt')
        write_metric_logs(epoch, metric_logger, writer)
        if epoch == 0:
            best_loss = losses;
        if best_loss >= losses:
            best_loss = losses
            if os.path.exists(f'{wts_path}/best.pt'):
                os.remove(f'{wts_path}/best.pt')
            torch.save(model.state_dict(), f'{wts_path}/best.pt')

def get_transform(is_train, **kwargs):
    default_params = {
        'augmentation': 'multiscale',
        'backend': 'tensor',
        'use_v2': False,

    }
    # 更新默认参数
    for key, value in kwargs.items():
        if key in default_params:
            default_params[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    # 解析参数
    augmentation = default_params['augmentation']
    backend = default_params['backend']
    use_v2 = default_params['use_v2']
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=augmentation, backend=backend, use_v2=use_v2
        )
    # elif weights and test_only:
    #     weights = torchvision.models.get_weight(args.weights)
    #     trans = weights.transforms()
    #     return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)


def write_metric_logs(epoch, metric_logger, writer):
    writer.add_scalar(f'loss_classifier:', metric_logger.meters['loss_classifier'].global_avg, epoch)
    writer.add_scalar(f'loss_box_reg:', metric_logger.meters['loss_box_reg'].global_avg, epoch)
    writer.add_scalar(f'loss_mask:', metric_logger.meters['loss_mask'].global_avg, epoch)
    writer.add_scalar(f'loss_objectness:', metric_logger.meters['loss_objectness'].global_avg, epoch)
    writer.add_scalar(f'loss_rpn_box_reg:', metric_logger.meters['loss_rpn_box_reg'].global_avg, epoch)
    writer.add_scalar(f'train loss:', metric_logger.meters['loss'].global_avg, epoch)