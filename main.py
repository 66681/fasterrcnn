import math
import os.path
import re
import sys

import PIL.Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.io import read_image
from pathlib import Path
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
import cv2
from sklearn.cluster import DBSCAN
from test.MaskRCNN import MaskRCNNDataset
from tools import utils
import pandas as pd

plt.rcParams["savefig.bbox"] = 'tight'
orig_path = r'F:\Downloads\severstal-steel-defect-detection'
dst_path = r'F:\Downloads\severstal-steel-defect-detection'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


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
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
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


def train():
    pass


def trans_datasets_format():
    # 使用pandas的read_csv函数读取文件
    df = pd.read_csv(os.path.join(orig_path, 'train.csv'))

    # 显示数据的前几行
    print(df.head())
    for row in df.itertuples():
        # print(f"Row index: {row.Index}")
        # print(getattr(row, 'ImageId'))  # 输出特定列的值
        img_name = getattr(row, 'ImageId')
        img_path = os.path.join(orig_path + '/train_images', img_name)
        dst_img_path = os.path.join(dst_path + '/images/train', img_name)
        dst_label_path = os.path.join(dst_path + '/labels/train', img_name[:-3] + 'txt')
        print(f'dst label:{dst_label_path}')
        im = cv2.imread(img_path)
        # cv2.imshow('val',im)
        cv2.imwrite(dst_img_path, im)
        img = PIL.Image.open(img_path)
        height, width = im.shape[:2]
        print(f'cv2 size:{im.shape}')
        label, mask = compute_mask(row, img.size)
        lbls, ins_masks=cluster_dbscan(mask,img)



        with open(dst_label_path, 'a+') as writer:
            # writer.write(label)
            for ins_mask in ins_masks:
                lbl_data = str(label) + ' '
                for mp in ins_mask:
                    h,w=mp
                    lbl_data += str(w / width) + ' ' + str(h / height) + ' '

                # non_zero_coords = np.nonzero(inm.reshape(width,height).T)
                # coords_list = list(zip(non_zero_coords[0], non_zero_coords[1]))
                # # print(f'mask:{mask[0,333]}')
                # print(f'mask pixels:{coords_list}')
                #
                #
                # for coord in coords_list:
                #     h, w = coord
                #     lbl_data += str(w / width) + ' ' + str(h / height) + ' '

                writer.write(lbl_data + '\n')
                print(f'lbl_data:{lbl_data}')
        writer.close()
        print(f'label:{label}')
        # plt.imshow(img)
        # plt.imshow(mask, cmap='Reds', alpha=0.3)
        # plt.show()


def compute_mask(row, shape):
    width, height = shape
    print(f'shape:{shape}')
    mask = np.zeros(width * height, dtype=np.uint8)
    pixels = np.array(list(map(int, row.EncodedPixels.split())))
    label = row.ClassId
    # print(f'pixels:{pixels}')
    mask_start = pixels[0::2]
    mask_length = pixels[1::2]

    for s, l in zip(mask_start, mask_length):
        mask[s:s + l] = 255
    mask = mask.reshape((width, height)).T

    # mask = np.flipud(np.rot90(mask.reshape((height, width))))
    return label, mask

def cluster_dbscan(mask,image):
    # 将 mask 转换为二值图像
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 将 mask 一维化
    mask_flattened = mask_binary.flatten()

    # 获取 mask 中的前景像素坐标
    foreground_pixels = np.argwhere(mask_flattened == 255)

    # 将像素坐标转换为二维坐标
    foreground_pixels_2d = np.column_stack(
        (foreground_pixels // mask_binary.shape[1], foreground_pixels % mask_binary.shape[1]))

    # 定义 DBSCAN 参数
    eps = 3  # 邻域半径
    min_samples = 10  # 最少样本数量

    # 应用 DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(foreground_pixels_2d)

    # 获取聚类标签
    labels = dbscan.labels_
    print(f'labels:{labels}')
    # 获取唯一的标签
    unique_labels = set(labels)

    print(f'unique_labels:{unique_labels}')
    # 创建一个空的图像来保存聚类结果
    clustered_image = np.zeros_like(image)
    # print(f'clustered_image shape:{clustered_image.shape}')


    # 将每个像素分配给相应的簇
    clustered_points=[]
    for k in unique_labels:


        class_member_mask = (labels == k)
        # print(f'class_member_mask:{class_member_mask}')
        # plt.subplot(132), plt.imshow(class_member_mask), plt.title(str(labels))

        pixel_indices = foreground_pixels_2d[class_member_mask]
        clustered_points.append(pixel_indices)

    return unique_labels,clustered_points

def show_cluster_dbscan(mask,image,unique_labels,clustered_points,):
    print(f'mask shape:{mask.shape}')
    # 将 mask 转换为二值图像
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 将 mask 一维化
    mask_flattened = mask_binary.flatten()

    # 获取 mask 中的前景像素坐标
    foreground_pixels = np.argwhere(mask_flattened == 255)
    # print(f'unique_labels:{unique_labels}')
    # 创建一个空的图像来保存聚类结果
    print(f'image shape:{image.shape}')
    clustered_image = np.zeros_like(image)
    print(f'clustered_image shape:{clustered_image.shape}')

    # 为每个簇分配颜色
    colors =np.array( [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))])
    # print(f'colors:{colors}')
    plt.figure(figsize=(12, 6))
    for points_coord,col in  zip(clustered_points,colors):
        for coord in points_coord:

            clustered_image[coord[0], coord[1]] = (np.array(col[:3]) * 255)

    # # 将每个像素分配给相应的簇
    # for k, col in zip(unique_labels, colors):
    #     print(f'col:{col*255}')
    #     if k == -1:
    #         # 黑色用于噪声点
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #     # print(f'class_member_mask:{class_member_mask}')
    #     # plt.subplot(132), plt.imshow(class_member_mask), plt.title(str(labels))
    #
    #     pixel_indices = foreground_pixels_2d[class_member_mask]
    #     clustered_points.append(pixel_indices)
    #     # print(f'pixel_indices:{pixel_indices}')
    #     for pixel_index in pixel_indices:
    #         clustered_image[pixel_index[0], pixel_index[1]] = (np.array(col[:3]) * 255)

    print(f'clustered_points:{len(clustered_points)}')
    # print(f'clustered_image:{clustered_image}')
    # 显示原图和聚类结果
    # plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(image), plt.title('Original Image')
    # print(f'image:{image}')
    plt.subplot(132), plt.imshow(mask_binary, cmap='gray'), plt.title('Mask')
    plt.subplot(133), plt.imshow(clustered_image.astype(np.uint8)), plt.title('Clustered Image')
    plt.show()
def test():
    dog1_int = read_image(str(Path('./assets') / 'dog1.jpg'))
    dog2_int = read_image(str(Path('./assets') / 'dog2.jpg'))
    dog_list = [dog1_int, dog2_int]
    grid = make_grid(dog_list)

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    transforms = weights.transforms()

    images = [transforms(d) for d in dog_list]
    # 假设输入图像的尺寸为 (3, 800, 800)
    dummy_input = torch.randn(1, 3, 800, 800)
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
    model = model.eval()

    # 使用 torch.jit.script
    scripted_model = torch.jit.script(model)

    output = model(dummy_input)
    print(f'output:{output}')

    writer = SummaryWriter('runs/')
    writer.add_graph(scripted_model, input_to_model=dummy_input)
    writer.flush()

    # torch.onnx.export(models,images, f='maskrcnn.onnx')  # 导出 .onnx 文
    # netron.start('AlexNet.onnx')  # 展示结构图

    show(grid)


def test_mask():
    name = 'fdb7c0397'
    label_path = os.path.join(dst_path + '/labels/train', name + '.txt')
    img_path = os.path.join(orig_path + '/train_images', name + '.jpg')
    mask = np.zeros((256, 1600), dtype=np.uint8)
    df = pd.read_csv(os.path.join(orig_path, 'train.csv'))
    # 显示数据的前几行
    print(df.head())
    points = []
    with open(label_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            parts = line.strip().split()
            # print(f'parts:{parts}')
            class_id = int(parts[0])
            x_array = parts[1::2]
            y_array = parts[2::2]

            for x, y in zip(x_array, y_array):
                x = float(x)
                y = float(y)
                points.append((int(y * 255), int(x * 1600)))
            # points = np.array([[float(parts[i]), float(parts[i + 1])] for i in range(1, len(parts), 2)])
            # mask_resized = cv2.resize(points, (1600, 256), interpolation=cv2.INTER_NEAREST)
            print(f'points:{points}')
            # mask[points[:,0],points[:,1]]=255
            for p in points:
                mask[p] = 255
            # cv2.fillPoly(mask, points, color=(255,))
    cv2.imshow('mask', mask)
    for row in df.itertuples():
        img_name = name + '.jpg'
        if img_name == getattr(row, 'ImageId'):
            img = PIL.Image.open(img_path)
            height, width = img.size
            print(f'img size:{img.size}')
            label, mask = compute_mask(row, img.size)
            plt.imshow(img)
            plt.imshow(mask, cmap='Reds', alpha=0.3)
            plt.show()
    cv2.waitKey(0)

def show_img_mask(img_path):
    test_img = PIL.Image.open(img_path)

    w,h=test_img.size
    test_img=torchvision.transforms.ToTensor()(test_img)
    test_img=test_img.permute(1, 2, 0)
    print(f'test_img shape:{test_img.shape}')
    lbl_path=re.sub(r'\\images\\', r'\\labels\\', img_path[:-3]) + 'txt'
    # print(f'lbl_path:{lbl_path}')
    masks = []
    labels = []

    with open(lbl_path, 'r') as reader:
        lines = reader.readlines()
        # 为每个簇分配颜色
        colors = np.array([plt.cm.Spectral(each) for each in np.linspace(0, 1, len(lines))])
        print(f'colors:{colors*255}')
        mask_points = []
        for line ,col in zip(lines,colors):
            print(f'col:{np.array(col[:3]) * 255}')
            mask = torch.zeros(test_img.shape, dtype=torch.uint8)
            # print(f'mask shape:{mask.shape}')
            parts = line.strip().split()
            # print(f'parts:{parts}')
            cls = torch.tensor(int(parts[0]), dtype=torch.int64)
            labels.append(cls)
            x_array = parts[1::2]
            y_array = parts[2::2]

            for x, y in zip(x_array, y_array):
                x = float(x)
                y = float(y)
                mask_points.append((int(y * h), int(x * w)))
            for p in mask_points:
                # print(f'p:{p}')
                mask[p] = torch.tensor(np.array(col[:3])*255)
            masks.append(mask)
    reader.close()
    target = {}

    # target["boxes"] = masks_to_boxes(torch.stack(masks))

    # target["labels"] = torch.stack(labels)

    target["masks"] = torch.stack(masks)
    print(f'target:{target}')

    # plt.imshow(test_img.permute(1, 2, 0))
    fig, axs = plt.subplots(2, 1)
    print(f'test_img:{test_img*255}')
    axs[0].imshow(test_img)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[1].imshow(test_img*255)
    for img_mask in target['masks']:
        # img_mask=img_mask.unsqueeze(0)
        # img_mask = img_mask.expand_as(test_img)
        # print(f'img_mask:{img_mask.shape}')
        axs[1].imshow(img_mask,alpha=0.3)

        # img_mask=np.array(img_mask)
        # print(f'img_mask:{img_mask.shape}')
        # plt.imshow(img_mask,alpha=0.5)
        # mask_3channel = cv2.merge([np.zeros_like(img_mask), np.zeros_like(img_mask), img_mask])
        # masked_image = cv2.addWeighted(test_img, 1, mask_3channel, 0.6, 0)

    # cv2.imshow('cv2 mask img', masked_image)
    # cv2.waitKey(0)
    plt.show()
def show_dataset():
    global transforms, dataset, imgs
    transforms = v2.Compose([
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        # v2.RandomPhotometricDistort(p=1),
        # v2.RandomHorizontalFlip(p=1),
        v2.ToTensor()
    ])
    dataset = MaskRCNNDataset(dataset_path=r'F:\Downloads\severstal-steel-defect-detection', transforms=transforms,
                              dataset_type='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)
    imgs, targets = next(iter(dataloader))

    mask = np.array(targets[2]['masks'][0])
    boxes = targets[2]['boxes']
    print(f'boxes:{boxes}')
    # mask[mask == 255] = 1
    img = np.array(imgs[2].permute(1, 2, 0)) * 255
    img = img.astype(np.uint8)
    print(f'img shape:{img.shape}')
    print(f'mask:{mask.shape}')
    # print(f'target:{targets}')
    # print(f'imgs:{imgs[0]}')
    # print(f'cv2 img shape:{np.array(imgs[0]).shape}')
    # cv2.imshow('cv2 img',img)
    # cv2.imshow('cv2 mask', mask)
    # plt.imshow('mask',mask)
    mask_3channel = cv2.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
    # cv2.imshow('mask_3channel',mask_3channel)
    print(f'mask_3channel:{mask_3channel.shape}')
    masked_image = cv2.addWeighted(img, 1, mask_3channel, 0.6, 0)
    # cv2.imshow('cv2 mask img', masked_image)
    plt.imshow(imgs[0].permute(1, 2, 0))
    plt.imshow(mask, cmap='Reds', alpha=0.3)
    drawn_boxes = draw_bounding_boxes((imgs[2] * 255).to(torch.uint8), boxes, colors="red", width=5)
    plt.imshow(drawn_boxes.permute(1, 2, 0))
    # show(drawn_boxes)
    plt.show()
    cv2.waitKey(0)

def test_cluster(img_path):
    test_img = PIL.Image.open(img_path)
    w, h = test_img.size
    test_img = torchvision.transforms.ToTensor()(test_img)
    test_img=(test_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # print(f'test_img:{test_img}')
    lbl_path = re.sub(r'\\images\\', r'\\labels\\', img_path[:-3]) + 'txt'
    # print(f'lbl_path:{lbl_path}')
    masks = []
    labels = []
    with open(lbl_path, 'r') as reader:
        lines = reader.readlines()
        mask_points = []
        for line in lines:
            mask = torch.zeros((h, w), dtype=torch.uint8)
            parts = line.strip().split()
            # print(f'parts:{parts}')
            cls = torch.tensor(int(parts[0]), dtype=torch.int64)
            labels.append(cls)
            x_array = parts[1::2]
            y_array = parts[2::2]

            for x, y in zip(x_array, y_array):
                x = float(x)
                y = float(y)
                mask_points.append((int(y * h), int(x * w)))
            for p in mask_points:
                mask[p] = 255
            masks.append(mask)
    # print(f'masks:{masks}')
    labels,clustered_points=cluster_dbscan(masks[0].numpy(),test_img)
    print(f'labels:{labels}')
    print(f'clustered_points len:{len(clustered_points)}')
    show_cluster_dbscan(masks[0].numpy(),test_img,labels,clustered_points)

if __name__ == '__main__':
    # trans_datasets_format()
    # test_mask()
    # 定义转换
    # show_dataset()

    # test_img_path= r"F:\Downloads\severstal-steel-defect-detection\images\train\0025bde0c.jpg"
    test_img_path = r"F:\DevTools\datasets\renyaun\1012\spilt\images\train\2024-09-27-14-32-53_SaveImage.png"
    # test_img1_path=r"F:\Downloads\severstal-steel-defect-detection\images\train\1d00226a0.jpg"
    show_img_mask(test_img_path)
    #
    # test_cluster(test_img_path)
