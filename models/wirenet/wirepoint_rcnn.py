import os
from typing import Optional, Any

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
# from torchinfo import summary
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor, \
    KeypointRCNN_ResNet50_FPN_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
# from visdom import Visdom

from models.config import config_tool
from models.config.config_tool import read_yaml
from models.ins.trainer import get_transform
from models.wirenet.head import RoIHeads
from models.wirenet.wirepoint_dataset import WirePointDataset
from tools import utils

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import io
import os.path as osp
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from models.wirenet.postprocess import postprocess

FEATURE_DIM = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()

        planes = outplanes // 2
        self.op = nn.Sequential(
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, outplanes, kernel_size=1),
        )

    def forward(self, x):
        return x + self.op(x)


class WirepointRCNN(FasterRCNN):
    def __init__(
            self,
            backbone,
            num_classes=None,
            # transform parameters
            min_size=None,
            max_size=1333,
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # keypoint parameters
            keypoint_roi_pool=None,
            keypoint_head=None,
            keypoint_predictor=None,
            num_keypoints=None,
            wirepoint_roi_pool=None,
            wirepoint_head=None,
            wirepoint_predictor=None,
            **kwargs,
    ):
        if not isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                "keypoint_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(keypoint_roi_pool)}"
            )
        if min_size is None:
            min_size = (640, 672, 704, 736, 768, 800)

        if num_keypoints is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_keypoints should be None when keypoint_predictor is specified")
        else:
            num_keypoints = 17

        out_channels = backbone.out_channels

        if wirepoint_roi_pool is None:
            wirepoint_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=128,
                                                    sampling_ratio=2, )

        if wirepoint_head is None:
            keypoint_layers = tuple(512 for _ in range(8))
            # print(f'keypoinyrcnnHeads inchannels:{out_channels},layers{keypoint_layers}')
            wirepoint_head = WirepointHead(out_channels, keypoint_layers)

        if wirepoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            wirepoint_predictor = WirepointPredictor()

        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            **kwargs,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            # wirepoint_roi_pool=wirepoint_roi_pool,
            # wirepoint_head=wirepoint_head,
            # wirepoint_predictor=wirepoint_predictor,
        )
        self.roi_heads = roi_heads

        self.roi_heads.wirepoint_roi_pool = wirepoint_roi_pool
        self.roi_heads.wirepoint_head = wirepoint_head
        self.roi_heads.wirepoint_predictor = wirepoint_predictor


class WirepointHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(WirepointHead, self).__init__()
        self.head_size = [[2], [1], [2]]
        m = int(input_channels / 4)
        heads = []
        # print(f'M.head_size:{M.head_size}')
        # for output_channels in sum(M.head_size, []):
        for output_channels in sum(self.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        # for idx, head in enumerate(self.heads):
        #     print(f'{idx},multitask head:{head(x).shape},input x:{x.shape}')

        outputs = torch.cat([head(x) for head in self.heads], dim=1)

        features = x
        return outputs, features


class WirepointPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        # self.backbone = backbone
        # self.cfg = read_yaml(cfg)
        self.cfg = read_yaml('wirenet.yaml')
        self.n_pts0 = self.cfg['model']['n_pts0']
        self.n_pts1 = self.cfg['model']['n_pts1']
        self.n_stc_posl = self.cfg['model']['n_stc_posl']
        self.dim_loi = self.cfg['model']['dim_loi']
        self.use_conv = self.cfg['model']['use_conv']
        self.dim_fc = self.cfg['model']['dim_fc']
        self.n_out_line = self.cfg['model']['n_out_line']
        self.n_out_junc = self.cfg['model']['n_out_junc']
        self.loss_weight = self.cfg['model']['loss_weight']
        self.n_dyn_junc = self.cfg['model']['n_dyn_junc']
        self.eval_junc_thres = self.cfg['model']['eval_junc_thres']
        self.n_dyn_posl = self.cfg['model']['n_dyn_posl']
        self.n_dyn_negl = self.cfg['model']['n_dyn_negl']
        self.n_dyn_othr = self.cfg['model']['n_dyn_othr']
        self.use_cood = self.cfg['model']['use_cood']
        self.use_slop = self.cfg['model']['use_slop']
        self.n_stc_negl = self.cfg['model']['n_stc_negl']
        self.head_size = self.cfg['model']['head_size']
        self.num_class = sum(sum(self.head_size, []))
        self.head_off = np.cumsum([sum(h) for h in self.head_size])

        lambda_ = torch.linspace(0, 1, self.n_pts0)[:, None]
        self.register_buffer("lambda_", lambda_)
        self.do_static_sampling = self.n_stc_posl + self.n_stc_negl > 0

        self.fc1 = nn.Conv2d(256, self.dim_loi, 1)
        scale_factor = self.n_pts0 // self.n_pts1
        if self.use_conv:
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(self.dim_loi, self.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(self.dim_loi * self.n_pts1 + FEATURE_DIM, 1)
            )
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(self.dim_loi * self.n_pts1 + FEATURE_DIM, self.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim_fc, self.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim_fc, 1),
            )
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, features, targets=None):

        # outputs, features = input
        # for out in outputs:
        #     print(f'out:{out.shape}')
        # outputs=merge_features(outputs,100)
        batch, channel, row, col = inputs.shape
        # print(f'outputs:{inputs.shape}')
        # print(f'batch:{batch}, channel:{channel}, row:{row}, col:{col}')

        if targets is not None:
            self.training = True
            # print(f'target:{targets}')
            wires_targets = [t["wires"] for t in targets]
            # print(f'wires_target:{wires_targets}')
            # 提取所有 'junc_map', 'junc_offset', 'line_map' 的张量
            junc_maps = [d["junc_map"] for d in wires_targets]
            junc_offsets = [d["junc_offset"] for d in wires_targets]
            line_maps = [d["line_map"] for d in wires_targets]

            junc_map_tensor = torch.stack(junc_maps, dim=0)
            junc_offset_tensor = torch.stack(junc_offsets, dim=0)
            line_map_tensor = torch.stack(line_maps, dim=0)

            wires_meta = {
                "junc_map": junc_map_tensor,
                "junc_offset": junc_offset_tensor,
                # "line_map": line_map_tensor,
            }
        else:
            self.training = False
            t = {
                "junc_coords": torch.zeros(1, 2).to(device),
                "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                "line_pos_idx": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                "line_neg_idx": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                "junc_map": torch.zeros([1, 1, 128, 128]).to(device),
                "junc_offset": torch.zeros([1, 1, 2, 128, 128]).to(device),
            }
            wires_targets = [t for b in range(inputs.size(0))]

            wires_meta = {
                "junc_map": torch.zeros([1, 1, 128, 128]).to(device),
                "junc_offset": torch.zeros([1, 1, 2, 128, 128]).to(device),
            }

        T = wires_meta.copy()
        n_jtyp = T["junc_map"].shape[1]
        offset = self.head_off
        result = {}
        for stack, output in enumerate([inputs]):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            # print(f"Stack {stack} output shape: {output.shape}")  # 打印每层的输出形状
            jmap = output[0: offset[0]].reshape(n_jtyp, 2, batch, row, col)
            lmap = output[offset[0]: offset[1]].squeeze(0)
            joff = output[offset[1]: offset[2]].reshape(n_jtyp, 2, batch, row, col)

            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }

        h = result["preds"]
        # print(f'features shape:{features.shape}')
        x = self.fc1(features)
        n_batch, n_channel, row, col = x.shape
        xs, ys, fs, ps, idx, jcs = [], [], [], [], [0], []

        for i, meta in enumerate(wires_targets):
            p, label, feat, jc = self.sample_lines(
                meta, h["jmap"][i], h["joff"][i],
            )
            # print(f"p.shape:{p.shape},label:{label.shape},feat:{feat.shape},jc:{len(jc)}")
            ys.append(label)
            if self.training and self.do_static_sampling:
                p = torch.cat([p, meta["lpre"]])
                feat = torch.cat([feat, meta["lpre_feat"]])
                ys.append(meta["lpre_label"])
                del jc
            else:
                jcs.append(jc)
                ps.append(p)
            fs.append(feat)

            p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
            p = p.reshape(-1, 2)  # [N_LINE x N_POINT, 2_XY]
            px, py = p[:, 0].contiguous(), p[:, 1].contiguous()
            px0 = px.floor().clamp(min=0, max=127)
            py0 = py.floor().clamp(min=0, max=127)
            px1 = (px0 + 1).clamp(min=0, max=127)
            py1 = (py0 + 1).clamp(min=0, max=127)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

            # xp: [N_LINE, N_CHANNEL, N_POINT]
            xp = (
                (
                        x[i, :, px0l, py0l] * (px1 - px) * (py1 - py)
                        + x[i, :, px1l, py0l] * (px - px0) * (py1 - py)
                        + x[i, :, px0l, py1l] * (px1 - px) * (py - py0)
                        + x[i, :, px1l, py1l] * (px - px0) * (py - py0)
                )
                    .reshape(n_channel, -1, self.n_pts0)
                    .permute(1, 0, 2)
            )
            xp = self.pooling(xp)
            # print(f'xp.shape:{xp.shape}')
            xs.append(xp)
            idx.append(idx[-1] + xp.shape[0])
            # print(f'idx__:{idx}')

        x, y = torch.cat(xs), torch.cat(ys)
        f = torch.cat(fs)
        x = x.reshape(-1, self.n_pts1 * self.dim_loi)
        x = torch.cat([x, f], 1)
        x = x.to(dtype=torch.float32)
        x = self.fc2(x).flatten()

        # return  x,idx,jcs,n_batch,ps,self.n_out_line,self.n_out_junc
        return x, y, idx, jcs, n_batch, ps, self.n_out_line, self.n_out_junc

        # if mode != "training":
        # self.inference(x, idx, jcs, n_batch, ps)

        # return result

    def sample_lines(self, meta, jmap, joff):
        with torch.no_grad():
            junc = meta["junc_coords"]  # [N, 2]
            jtyp = meta["jtyp"]  # [N]
            Lpos = meta["line_pos_idx"]
            Lneg = meta["line_neg_idx"]

            n_type = jmap.shape[0]
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            max_K = self.n_dyn_junc // n_type
            N = len(junc)
            # if mode != "training":
            if not self.training:
                K = min(int((jmap > self.eval_junc_thres).float().sum().item()), max_K)
            else:
                K = min(int(N * 2 + 2), max_K)
            if K < 2:
                K = 2
            device = jmap.device

            # index: [N_TYPE, K]
            score, index = torch.topk(jmap, k=K)
            y = (index // 128).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            x = (index % 128).float() + torch.gather(joff[:, 1], 1, index) + 0.5

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y, index

            # print(f"xy_.is_cuda: {xy_.is_cuda}")
            # print(f"junc.is_cuda: {junc.is_cuda}")

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            # xy: [N_TYPE * K, 2]
            # match: [N_TYPE, K]
            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]
            label = Lpos[up, vp]

            # if mode == "training":
            if self.training:
                c = torch.zeros_like(label, dtype=torch.bool)

                # sample positive lines
                cdx = label.nonzero().flatten()
                if len(cdx) > self.n_dyn_posl:
                    # print("too many positive lines")
                    perm = torch.randperm(len(cdx), device=device)[: self.n_dyn_posl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # sample negative lines
                cdx = Lneg[up, vp].nonzero().flatten()
                if len(cdx) > self.n_dyn_negl:
                    # print("too many negative lines")
                    perm = torch.randperm(len(cdx), device=device)[: self.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # sample other (unmatched) lines
                cdx = torch.randint(len(c), (self.n_dyn_othr,), device=device)
                c[cdx] = 1
            else:
                c = (u < v).flatten()

            # sample lines
            u, v, label = u[c], v[c], label[c]
            xy = xy.reshape(n_type * K, 2)
            xyu, xyv = xy[u], xy[v]

            u2v = xyu - xyv
            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)
            feat = torch.cat(
                [
                    xyu / 128 * self.use_cood,
                    xyv / 128 * self.use_cood,
                    u2v * self.use_slop,
                    (u[:, None] > K).float(),
                    (v[:, None] > K).float(),
                ],
                1,
            )
            line = torch.cat([xyu[:, None], xyv[:, None]], 1)

            xy = xy.reshape(n_type, K, 2)
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]
            return line, label.float(), feat, jcs


def wirepointrcnn_resnet50_fpn(
        *,
        weights: Optional[KeypointRCNN_ResNet50_FPN_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        num_keypoints: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> WirepointRCNN:
    weights = KeypointRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = WirepointRCNN(backbone, num_classes=5, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == KeypointRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model


def _loss(losses):
    total_loss = 0
    for i in losses.keys():
        if i != "loss_wirepoint":
            total_loss += losses[i]
        else:
            loss_labels = losses[i]["losses"]
    loss_labels_k = list(loss_labels[0].keys())
    for j, name in enumerate(loss_labels_k):
        loss = loss_labels[0][name].mean()
        total_loss += loss

    return total_loss


cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    plt.tight_layout()
    plt.imshow(im)
    plt.colorbar(sm, fraction=0.046)
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])
    # plt.show()


# def _plot_samples(img, i, result, prefix, epoch):
#     print(f"prefix:{prefix}")
#     def draw_vecl(lines, sline, juncs, junts, fn):
#         directory = os.path.dirname(fn)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         imshow(img.permute(1, 2, 0))
#         if len(lines) > 0 and not (lines[0] == 0).all():
#             for i, ((a, b), s) in enumerate(zip(lines, sline)):
#                 if i > 0 and (lines[i] == lines[0]).all():
#                     break
#                 plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=4)
#         if not (juncs[0] == 0).all():
#             for i, j in enumerate(juncs):
#                 if i > 0 and (i == juncs[0]).all():
#                     break
#                 plt.scatter(j[1], j[0], c="red", s=64, zorder=100)
#         if junts is not None and len(junts) > 0 and not (junts[0] == 0).all():
#             for i, j in enumerate(junts):
#                 if i > 0 and (i == junts[0]).all():
#                     break
#                 plt.scatter(j[1], j[0], c="blue", s=64, zorder=100)
#         plt.savefig(fn), plt.close()
#
#     rjuncs = result["juncs"][i].cpu().numpy() * 4
#     rjunts = None
#     if "junts" in result:
#         rjunts = result["junts"][i].cpu().numpy() * 4
#
#     vecl_result = result["lines"][i].cpu().numpy() * 4
#     score = result["score"][i].cpu().numpy()
#
#     draw_vecl(vecl_result, score, rjuncs, rjunts, f"{prefix}_vecl_b.jpg")
#
#     img1 = cv2.imread(f"{prefix}_vecl_b.jpg")
#     writer.add_image(f'output_epoch_{epoch}', img1, global_step=epoch)

def _plot_samples(img, i, result, prefix, epoch, writer):
    # print(f"prefix:{prefix}")

    def draw_vecl(lines, sline, juncs, junts, fn):
        # 确保目录存在
        directory = os.path.dirname(fn)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 绘制图像
        plt.figure()
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')  # 可选：关闭坐标轴

        if len(lines) > 0 and not (lines[0] == 0).all():
            for idx, ((a, b), s) in enumerate(zip(lines, sline)):
                if idx > 0 and (lines[idx] == lines[0]).all():
                    break
                plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=1)

        if not (juncs[0] == 0).all():
            for idx, j in enumerate(juncs):
                if idx > 0 and (j == juncs[0]).all():
                    break
                plt.scatter(j[1], j[0], c="red", s=20, zorder=100)

        if junts is not None and len(junts) > 0 and not (junts[0] == 0).all():
            for idx, j in enumerate(junts):
                if idx > 0 and (j == junts[0]).all():
                    break
                plt.scatter(j[1], j[0], c="blue", s=20, zorder=100)

        # plt.show()

        # 将matplotlib图像转换为numpy数组
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return image_from_plot

    # 获取结果数据并转换为numpy数组
    rjuncs = result["juncs"][i].cpu().numpy() * 4
    rjunts = None
    if "junts" in result:
        rjunts = result["junts"][i].cpu().numpy() * 4

    vecl_result = result["lines"][i].cpu().numpy() * 4
    score = result["score"][i].cpu().numpy()

    # 调用绘图函数并获取图像
    image_path = f"{prefix}_vecl_b.jpg"
    image_array = draw_vecl(vecl_result, score, rjuncs, rjunts, image_path)

    # 将numpy数组转换为torch tensor，并写入TensorBoard
    image_tensor = transforms.ToTensor()(image_array)
    writer.add_image(f'output_epoch', image_tensor, global_step=epoch)
    writer.add_image(f'ori_epoch', img, global_step=epoch)


def show_line(img, pred, prefix, epoch, write):
    fn = f"{prefix}_line.jpg"
    directory = os.path.dirname(fn)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(fn)
    PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
    H = pred

    im = img.permute(1, 2, 0)

    lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
    scores = H["score"][0].cpu().numpy()
    for i in range(1, len(lines)):
        if (lines[i] == lines[0]).all():
            lines = lines[:i]
            scores = scores[:i]
            break

    # postprocess lines to remove overlapped lines
    diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
    nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

    for i, t in enumerate([0.5]):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        for (a, b), s in zip(nlines, nscores):
            if s < t:
                continue
            plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            plt.scatter(a[1], a[0], **PLTOPTS)
            plt.scatter(b[1], b[0], **PLTOPTS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(im)
        plt.savefig(fn, bbox_inches="tight")
        plt.show()
        plt.close()


        img2 = cv2.imread(fn)  # 预测图
        # img1 = im.resize(img2.shape)  # 原图

        # writer.add_images(f"{epoch}", torch.tensor([img1, img2]), dataformats='NHWC')
        writer.add_image("output", img2, epoch)


if __name__ == '__main__':
    cfg = 'wirenet.yaml'
    cfg = read_yaml(cfg)
    print(f'cfg:{cfg}')
    print(cfg['model']['n_dyn_negl'])
    # net = WirepointPredictor()

    # if torch.cuda.is_available():
    #     device_name = "cuda"
    #     torch.backends.cudnn.deterministic = True
    #     torch.cuda.manual_seed(0)
    #     print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    # else:
    #     print("CUDA is not available")
    #
    # device = torch.device(device_name)

    dataset_train = WirePointDataset(dataset_path=cfg['io']['datadir'], dataset_type='train')
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=1, drop_last=True)
    train_collate_fn = utils.collate_fn_wirepoint
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=train_batch_sampler, num_workers=0, collate_fn=train_collate_fn
    )

    dataset_val = WirePointDataset(dataset_path=cfg['io']['datadir'], dataset_type='val')
    val_sampler = torch.utils.data.RandomSampler(dataset_val)
    # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size=1, drop_last=True)
    val_collate_fn = utils.collate_fn_wirepoint
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_sampler=val_batch_sampler, num_workers=0, collate_fn=val_collate_fn
    )

    model = wirepointrcnn_resnet50_fpn().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['optim']['lr'])
    writer = SummaryWriter(cfg['io']['logdir'])


    def move_to_device(data, device):
        if isinstance(data, (list, tuple)):
            return type(data)(move_to_device(item, device) for item in data)
        elif isinstance(data, dict):
            return {key: move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data  # 对于非张量类型的数据不做任何改变


    def writer_loss(writer, losses, epoch):
        # ??????
        try:
            for key, value in losses.items():
                if key == 'loss_wirepoint':
                    # ?? wirepoint ??????
                    for subdict in losses['loss_wirepoint']['losses']:
                        for subkey, subvalue in subdict.items():
                            # ?? .item() ?????
                            writer.add_scalar(f'loss_wirepoint/{subkey}',
                                              subvalue.item() if hasattr(subvalue, 'item') else subvalue,
                                              epoch)
                elif isinstance(value, torch.Tensor):
                    # ????????
                    writer.add_scalar(key, value.item(), epoch)
        except Exception as e:
            print(f"TensorBoard logging error: {e}")


    for epoch in range(cfg['optim']['max_epoch']):
        print(f"epoch:{epoch}")
        model.train()

        for imgs, targets in data_loader_train:
            losses = model(move_to_device(imgs, device), move_to_device(targets, device))
            loss = _loss(losses)
            print(loss)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # writer_loss(writer, losses, epoch)

        # model.eval()
        # with torch.no_grad():
        #     for batch_idx, (imgs, targets) in enumerate(data_loader_val):
        #         pred = model(move_to_device(imgs, device))
        #         # print(f"pred:{pred}")
        #
        #         if batch_idx == 0:
        #             result = pred[1]['wires']  # pred[0].keys()   ['boxes', 'labels', 'scores']
        #             print(imgs[0].shape)  # [3，512，512]
        #             # imshow(imgs[0].permute(1, 2, 0))  # 改为(512, 512, 3)
        #             _plot_samples(imgs[0], 0, result, f"{cfg['io']['logdir']}/{epoch}/", epoch, writer)
                    # show_line(imgs[0], result, f"{cfg['io']['logdir']}/{epoch}", epoch, writer)

# imgs, targets = next(iter(data_loader))
#
# model.train()
# pred = model(imgs, targets)
# print(f'pred:{pred}')

# result, losses = model(imgs, targets)
# print(f'result:{result}')
# print(f'pred:{losses}')
