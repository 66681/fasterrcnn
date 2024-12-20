from torch.utils.data.dataset import T_co

from models.base.base_dataset import BaseDataset

import glob
import json
import math
import os
import random
import cv2
import PIL

import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision.utils import draw_bounding_boxes

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import matplotlib.pyplot as plt
from models.dataset_tool import line_boxes, read_masks_from_txt_wire, read_masks_from_pixels_wire, adjacency_matrix


class WirePointDataset(BaseDataset):
    def __init__(self, dataset_path, transforms=None, dataset_type=None, target_type='pixel'):
        super().__init__(dataset_path)

        self.data_path = dataset_path
        print(f'data_path:{dataset_path}')
        self.transforms = transforms
        self.img_path = os.path.join(dataset_path, "images\\" + dataset_type)
        self.lbl_path = os.path.join(dataset_path, "labels\\" + dataset_type)
        self.imgs = os.listdir(self.img_path)
        self.lbls = os.listdir(self.lbl_path)
        self.target_type = target_type
        # self.default_transform = DefaultTransform()

    def __getitem__(self, index) -> T_co:
        img_path = os.path.join(self.img_path, self.imgs[index])
        lbl_path = os.path.join(self.lbl_path, self.imgs[index][:-3] + 'json')

        img = PIL.Image.open(img_path).convert('RGB')
        w, h = img.size

        # wire_labels, target = self.read_target(item=index, lbl_path=lbl_path, shape=(h, w))
        target = self.read_target(item=index, lbl_path=lbl_path, shape=(h, w))
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = self.default_transform(img)

        # print(f'img:{img}')
        return img, target

    def __len__(self):
        return len(self.imgs)

    def read_target(self, item, lbl_path, shape, extra=None):
        # print(f'lbl_path:{lbl_path}')
        with open(lbl_path, 'r') as file:
            lable_all = json.load(file)

        n_stc_posl = 300
        n_stc_negl = 40
        use_cood = 0
        use_slop = 0

        wire = lable_all["wires"][0]  # 字典
        line_pos_coords = np.random.permutation(wire["line_pos_coords"]["content"])[: n_stc_posl]  # 不足，有多少取多少
        line_neg_coords = np.random.permutation(wire["line_neg_coords"]["content"])[: n_stc_negl]
        npos, nneg = len(line_pos_coords), len(line_neg_coords)
        lpre = np.concatenate([line_pos_coords, line_neg_coords], 0)  # 正负样本坐标合在一起
        for i in range(len(lpre)):
            if random.random() > 0.5:
                lpre[i] = lpre[i, ::-1]
        ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
        ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
        feat = [
            lpre[:, :, :2].reshape(-1, 4) / 128 * use_cood,
            ldir * use_slop,
            lpre[:, :, 2],
        ]
        feat = np.concatenate(feat, 1)

        wire_labels = {
            "junc_coords": torch.tensor(wire["junc_coords"]["content"])[:, :2],
            "jtyp": torch.tensor(wire["junc_coords"]["content"])[:, 2].byte(),
            "line_pos_idx": adjacency_matrix(len(wire["junc_coords"]["content"]), wire["line_pos_idx"]["content"]),
            # 真实存在线条的邻接矩阵
            "line_neg_idx": adjacency_matrix(len(wire["junc_coords"]["content"]), wire["line_neg_idx"]["content"]),
            # 不存在线条的临界矩阵
            "lpre": torch.tensor(lpre)[:, :, :2],
            "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),  # 样本对应标签 1，0
            "lpre_feat": torch.from_numpy(feat),
            "junc_map": torch.tensor(wire['junc_map']["content"]),
            "junc_offset": torch.tensor(wire['junc_offset']["content"]),
            "line_map": torch.tensor(wire['line_map']["content"]),
        }

        labels = []
        if self.target_type == 'polygon':
            labels, masks = read_masks_from_txt_wire(lbl_path, shape)
        elif self.target_type == 'pixel':
            labels = read_masks_from_pixels_wire(lbl_path, shape)

        # print(torch.stack(masks).shape)    # [线段数, 512, 512]
        target = {}
        target["labels"] = torch.stack(labels)
        target["image_id"] = torch.tensor(item)
        # return wire_labels, target
        target["wires"] = wire_labels
        target["boxes"] = line_boxes(target)
        return target

    def show(self, idx):
        image, target = self.__getitem__(idx)

        cmap = plt.get_cmap("jet")
        norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        def imshow(im):
            plt.close()
            plt.tight_layout()
            plt.imshow(im)
            plt.colorbar(sm, fraction=0.046)
            plt.xlim([0, im.shape[0]])
            plt.ylim([im.shape[0], 0])

        def draw_vecl(lines, sline, juncs, junts, fn=None):
            img_path = os.path.join(self.img_path, self.imgs[idx])
            imshow(io.imread(img_path))
            if len(lines) > 0 and not (lines[0] == 0).all():
                for i, ((a, b), s) in enumerate(zip(lines, sline)):
                    if i > 0 and (lines[i] == lines[0]).all():
                        break
                    plt.plot([a[1], b[1]], [a[0], b[0]], c="red", linewidth=1)  # a[1], b[1]无明确大小
            if not (juncs[0] == 0).all():
                for i, j in enumerate(juncs):
                    if i > 0 and (i == juncs[0]).all():
                        break
                    plt.scatter(j[1], j[0], c="red", s=2, zorder=100)  # 原 s=64


            img_path = os.path.join(self.img_path, self.imgs[idx])
            img = PIL.Image.open(img_path).convert('RGB')
            boxed_image = draw_bounding_boxes((self.default_transform(img) * 255).to(torch.uint8), target["boxes"],
                                              colors="yellow", width=1)
            plt.imshow(boxed_image.permute(1, 2, 0).numpy())
            plt.show()

            plt.show()
            if fn != None:
                plt.savefig(fn)

        junc = target['wires']['junc_coords'].cpu().numpy() * 4
        jtyp = target['wires']['jtyp'].cpu().numpy()
        juncs = junc[jtyp == 0]
        junts = junc[jtyp == 1]

        lpre = target['wires']["lpre"].cpu().numpy() * 4
        vecl_target = target['wires']["lpre_label"].cpu().numpy()
        lpre = lpre[vecl_target == 1]

        # draw_vecl(lpre, np.ones(lpre.shape[0]), juncs, junts, save_path)
        draw_vecl(lpre, np.ones(lpre.shape[0]), juncs, junts)


    def show_img(self, img_path):
        pass



