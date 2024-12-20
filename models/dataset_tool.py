import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import tools.transforms as reference_transforms
from collections import defaultdict

from tools import presets

import json


def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors

        return torchvision.transforms.v2, torchvision.tv_tensors
    else:
        return reference_transforms, None


class Augmentation:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
            self,
            *,
            data_augmentation,
            hflip_prob=0.5,
            mean=(123.0, 117.0, 104.0),
            backend="pil",
            use_v2=False,
    ):

        T, tv_tensors = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "lsj":
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                # TODO: FixedSizeCrop below doesn't work on tensors!
                reference_transforms.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0}) if use_v2 else list(mean)
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2.
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [
                T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                T.SanitizeBoundingBoxes(),
                T.ToPureTensor(),
            ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


def read_polygon_points(lbl_path, shape):
    """读取 YOLOv8 格式的标注文件并解析多边形轮廓"""
    polygon_points = []
    w, h = shape[:2]
    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)  # 读取点坐标
        points[:, 0] *= h
        points[:, 1] *= w

        polygon_points.append((class_id, points))

    return polygon_points


def read_masks_from_pixels(lbl_path, shape):
    """读取纯像素点格式的文件，不是轮廓像素点"""
    h, w = shape
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
                mask[p] = 1
            masks.append(mask)
    reader.close()
    return labels, masks


def create_masks_from_polygons(polygons, image_shape):
    """创建一个与图像尺寸相同的掩码，并填充多边形轮廓"""
    colors = np.array([plt.cm.Spectral(each) for each in np.linspace(0, 1, len(polygons))])
    masks = []

    for polygon_data, col in zip(polygons, colors):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        # 将多边形顶点转换为 NumPy 数组
        _, polygon = polygon_data
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

        # 使用 OpenCV 的 fillPoly 函数填充多边形
        # print(f'color:{col[:3]}')
        cv2.fillPoly(mask, [pts], np.array(col[:3]) * 255)
        mask = torch.from_numpy(mask)
        mask[mask != 0] = 1
        masks.append(mask)

    return masks


def read_masks_from_txt(label_path, shape):
    polygon_points = read_polygon_points(label_path, shape)
    masks = create_masks_from_polygons(polygon_points, shape)
    labels = [torch.tensor(item[0]) for item in polygon_points]

    return labels, masks


def masks_to_boxes(masks: torch.Tensor, ) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(masks_to_boxes)
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)
        # debug to pixel datasets

        if bounding_boxes[index, 0] == bounding_boxes[index, 2]:
            bounding_boxes[index, 2] = bounding_boxes[index, 2] + 1
            bounding_boxes[index, 0] = bounding_boxes[index, 0] - 1

        if bounding_boxes[index, 1] == bounding_boxes[index, 3]:
            bounding_boxes[index, 3] = bounding_boxes[index, 3] + 1
            bounding_boxes[index, 1] = bounding_boxes[index, 1] - 1

    return bounding_boxes


def line_boxes(target):
    boxs = []
    lpre = target['wires']["lpre"].cpu().numpy() * 4
    vecl_target = target['wires']["lpre_label"].cpu().numpy()
    lpre = lpre[vecl_target == 1]

    lines = lpre
    sline = np.ones(lpre.shape[0])

    if len(lines) > 0 and not (lines[0] == 0).all():
        for i, ((a, b), s) in enumerate(zip(lines, sline)):
            if i > 0 and (lines[i] == lines[0]).all():
                break
            # plt.plot([a[1], b[1]], [a[0], b[0]], c="red", linewidth=1)  # a[1], b[1]无明确大小
            if a[1] > b[1]:
                ymax = a[1] + 1
                ymin = b[1] - 1
            else:
                ymin = a[1] - 1
                ymax = b[1] + 1
            if a[0] > b[0]:
                xmax = a[0] + 1
                xmin = b[0] - 1
            else:
                xmin = a[0] - 1
                xmax = b[0] + 1
            boxs.append([ymin, xmin, ymax, xmax])

    return torch.tensor(boxs)


def read_polygon_points_wire(lbl_path, shape):
    """读取 YOLOv8 格式的标注文件并解析多边形轮廓"""
    polygon_points = []
    w, h = shape[:2]
    with open(lbl_path, 'r') as f:
        lines = json.load(f)

    for line in lines["segmentations"]:
        parts = line["data"]
        class_id = int(line["cls_id"])
        points = np.array(parts, dtype=np.float32).reshape(-1, 2)  # 读取点坐标
        points[:, 0] *= h
        points[:, 1] *= w

        polygon_points.append((class_id, points))

    return polygon_points


def read_masks_from_txt_wire(label_path, shape):
    polygon_points = read_polygon_points_wire(label_path, shape)
    masks = create_masks_from_polygons(polygon_points, shape)
    labels = [torch.tensor(item[0]) for item in polygon_points]

    return labels, masks


def read_masks_from_pixels_wire(lbl_path, shape):
    """读取纯像素点格式的文件，不是轮廓像素点"""
    h, w = shape
    masks = []
    labels = []

    with open(lbl_path, 'r') as reader:
        lines = json.load(reader)
        mask_points = []
        for line in lines["segmentations"]:
            # mask = torch.zeros((h, w), dtype=torch.uint8)
            # parts = line["data"]
            # print(f'parts:{parts}')
            cls = torch.tensor(int(line["cls_id"]), dtype=torch.int64)
            labels.append(cls)
            # x_array = parts[0::2]
            # y_array = parts[1::2]
            # 
            # for x, y in zip(x_array, y_array):
            #     x = float(x)
            #     y = float(y)
            #     mask_points.append((int(y * h), int(x * w)))

            # for p in mask_points:
            #     mask[p] = 1
            # masks.append(mask)
    reader.close()
    return labels


def adjacency_matrix(n, link):  # 邻接矩阵
    mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
    link = torch.tensor(link)
    if len(link) > 0:
        mat[link[:, 0], link[:, 1]] = 1
        mat[link[:, 1], link[:, 0]] = 1
    return mat
