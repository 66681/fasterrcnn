from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from . import _utils as det_utils

from torch.utils.data.dataloader import default_collate


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


# def wirepoint_loss(target, outputs, feature, loss_weight,mode):
#     wires = target['wires']
#     result = {"feature": feature}
#     batch, channel, row, col = outputs[0].shape
#     print(f"Initial Output[0] shape: {outputs[0].shape}")  # 打印初始输出形状
#     print(f"Total Stacks: {len(outputs)}")  # 打印堆栈数
#
#     T = wires.copy()
#     n_jtyp = T["junc_map"].shape[1]
#     for task in ["junc_map"]:
#         T[task] = T[task].permute(1, 0, 2, 3)
#     for task in ["junc_offset"]:
#         T[task] = T[task].permute(1, 2, 0, 3, 4)
#
#     offset = self.head_off
#     loss_weight = loss_weight
#     losses = []
#
#     for stack, output in enumerate(outputs):
#         output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
#         print(f"Stack {stack} output shape: {output.shape}")  # 打印每层的输出形状
#         jmap = output[0: offset[0]].reshape(n_jtyp, 2, batch, row, col)
#         lmap = output[offset[0]: offset[1]].squeeze(0)
#         joff = output[offset[1]: offset[2]].reshape(n_jtyp, 2, batch, row, col)
#
#         if stack == 0:
#             result["preds"] = {
#                 "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
#                 "lmap": lmap.sigmoid(),
#                 "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
#             }
#             # visualize_feature_map(jmap[0, 0], title=f"jmap - Stack {stack}")
#             # visualize_feature_map(lmap, title=f"lmap - Stack {stack}")
#             # visualize_feature_map(joff[0, 0], title=f"joff - Stack {stack}")
#
#             if mode == "testing":
#                 return result
#
#         L = OrderedDict()
#         L["junc_map"] = sum(
#             cross_entropy_loss(jmap[i], T["junc_map"][i]) for i in range(n_jtyp)
#         )
#         L["line_map"] = (
#             F.binary_cross_entropy_with_logits(lmap, T["line_map"], reduction="none")
#             .mean(2)
#             .mean(1)
#         )
#         L["junc_offset"] = sum(
#             sigmoid_l1_loss(joff[i, j], T["junc_offset"][i, j], -0.5, T["junc_map"][i])
#             for i in range(n_jtyp)
#             for j in range(2)
#         )
#         for loss_name in L:
#             L[loss_name].mul_(loss_weight[loss_name])
#         losses.append(L)
#
#     result["losses"] = losses
#     return result

def wirepoint_head_line_loss(targets, output, x, y, idx, loss_weight):
    # output, feature: head返回结果
    # x, y, idx : line中间生成结果
    result = {}
    batch, channel, row, col = output.shape

    wires_targets = [t["wires"] for t in targets]
    wires_targets = wires_targets.copy()
    # print(f'wires_target:{wires_targets}')
    # 提取所有 'junc_map', 'junc_offset', 'line_map' 的张量
    junc_maps = [d["junc_map"] for d in wires_targets]
    junc_offsets = [d["junc_offset"] for d in wires_targets]
    line_maps = [d["line_map"] for d in wires_targets]

    junc_map_tensor = torch.stack(junc_maps, dim=0)
    junc_offset_tensor = torch.stack(junc_offsets, dim=0)
    line_map_tensor = torch.stack(line_maps, dim=0)
    T = {"junc_map": junc_map_tensor, "junc_offset": junc_offset_tensor, "line_map": line_map_tensor}

    n_jtyp = T["junc_map"].shape[1]

    for task in ["junc_map"]:
        T[task] = T[task].permute(1, 0, 2, 3)
    for task in ["junc_offset"]:
        T[task] = T[task].permute(1, 2, 0, 3, 4)

    offset = [2, 3, 5]
    losses = []
    output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
    jmap = output[0: offset[0]].reshape(n_jtyp, 2, batch, row, col)
    lmap = output[offset[0]: offset[1]].squeeze(0)
    joff = output[offset[1]: offset[2]].reshape(n_jtyp, 2, batch, row, col)
    L = OrderedDict()
    L["junc_map"] = sum(
        cross_entropy_loss(jmap[i], T["junc_map"][i]) for i in range(n_jtyp)
    )
    L["line_map"] = (
        F.binary_cross_entropy_with_logits(lmap, T["line_map"], reduction="none")
            .mean(2)
            .mean(1)
    )
    L["junc_offset"] = sum(
        sigmoid_l1_loss(joff[i, j], T["junc_offset"][i, j], -0.5, T["junc_map"][i])
        for i in range(n_jtyp)
        for j in range(2)
    )
    for loss_name in L:
        L[loss_name].mul_(loss_weight[loss_name])
    losses.append(L)
    result["losses"] = losses

    loss = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss(x, y)
    lpos_mask, lneg_mask = y, 1 - y
    loss_lpos, loss_lneg = loss * lpos_mask, loss * lneg_mask

    def sum_batch(x):
        xs = [x[idx[i]: idx[i + 1]].sum()[None] for i in range(batch)]
        return torch.cat(xs)

    lpos = sum_batch(loss_lpos) / sum_batch(lpos_mask).clamp(min=1)
    lneg = sum_batch(loss_lneg) / sum_batch(lneg_mask).clamp(min=1)
    result["losses"][0]["lpos"] = lpos * loss_weight["lpos"]
    result["losses"][0]["lneg"] = lneg * loss_weight["lneg"]

    return result


def wirepoint_inference(input, idx, jcs, n_batch, ps, n_out_line, n_out_junc):
    result = {}
    result["wires"] = {}
    p = torch.cat(ps)
    s = torch.sigmoid(input)
    b = s > 0.5
    lines = []
    score = []
    # print(f"n_batch:{n_batch}")
    for i in range(n_batch):
        # print(f"idx:{idx}")
        p0 = p[idx[i]: idx[i + 1]]
        s0 = s[idx[i]: idx[i + 1]]
        mask = b[idx[i]: idx[i + 1]]
        p0 = p0[mask]
        s0 = s0[mask]
        if len(p0) == 0:
            lines.append(torch.zeros([1, n_out_line, 2, 2], device=p.device))
            score.append(torch.zeros([1, n_out_line], device=p.device))
        else:
            arg = torch.argsort(s0, descending=True)
            p0, s0 = p0[arg], s0[arg]
            lines.append(p0[None, torch.arange(n_out_line) % len(p0)])
            score.append(s0[None, torch.arange(n_out_line) % len(s0)])
        for j in range(len(jcs[i])):
            if len(jcs[i][j]) == 0:
                jcs[i][j] = torch.zeros([n_out_junc, 2], device=p.device)
            jcs[i][j] = jcs[i][j][
                None, torch.arange(n_out_junc) % len(jcs[i][j])
            ]
    result["wires"]["lines"] = torch.cat(lines)
    result["wires"]["score"] = torch.cat(score)
    result["wires"]["juncs"] = torch.cat([jcs[i][0] for i in range(n_batch)])

    if len(jcs[i]) > 1:
        result["preds"]["junts"] = torch.cat(
            [jcs[i][1] for i in range(n_batch)]
        )

    return result


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    # print(f'mask_logits:{mask_logits},gt_masks:{gt_masks},,gt_labels:{gt_labels}]')
    # print(f'mask discretization_size:{discretization_size}')
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    # print(f'mask labels:{labels}')
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    # print(f'mask labels1:{labels}')
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    # print(f'mask_targets:{mask_targets.shape},mask_logits:{mask_logits.shape}')
    # print(f'mask_targets:{mask_targets}')
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    # print(f'mask_loss:{mask_loss}')
    return mask_loss


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


def _onnx_heatmaps_to_keypoints(
        maps, maps_i, roi_map_width, roi_map_height, widths_i, heights_i, offset_x_i, offset_y_i
):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)

    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height

    roi_map = F.interpolate(
        maps_i[:, None], size=(int(roi_map_height), int(roi_map_width)), mode="bicubic", align_corners=False
    )[:, 0]

    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

    x_int = pos % w
    y_int = (pos - x_int) // w

    x = (torch.tensor(0.5, dtype=torch.float32) + x_int.to(dtype=torch.float32)) * width_correction.to(
        dtype=torch.float32
    )
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int.to(dtype=torch.float32)) * height_correction.to(
        dtype=torch.float32
    )

    xy_preds_i_0 = x + offset_x_i.to(dtype=torch.float32)
    xy_preds_i_1 = y + offset_y_i.to(dtype=torch.float32)
    xy_preds_i_2 = torch.ones(xy_preds_i_1.shape, dtype=torch.float32)
    xy_preds_i = torch.stack(
        [
            xy_preds_i_0.to(dtype=torch.float32),
            xy_preds_i_1.to(dtype=torch.float32),
            xy_preds_i_2.to(dtype=torch.float32),
        ],
        0,
    )

    # TODO: simplify when indexing without rank will be supported by ONNX
    base = num_keypoints * num_keypoints + num_keypoints + 1
    ind = torch.arange(num_keypoints)
    ind = ind.to(dtype=torch.int64) * base
    end_scores_i = (
        roi_map.index_select(1, y_int.to(dtype=torch.int64))
            .index_select(2, x_int.to(dtype=torch.int64))
            .view(-1)
            .index_select(0, ind.to(dtype=torch.int64))
    )

    return xy_preds_i, end_scores_i


@torch.jit._script_if_tracing
def _onnx_heatmaps_to_keypoints_loop(
        maps, rois, widths_ceil, heights_ceil, widths, heights, offset_x, offset_y, num_keypoints
):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)

    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(
            maps, maps[i], widths_ceil[i], heights_ceil[i], widths[i], heights[i], offset_x[i], offset_y[i]
        )
        xy_preds = torch.cat((xy_preds.to(dtype=torch.float32), xy_preds_i.unsqueeze(0).to(dtype=torch.float32)), 0)
        end_scores = torch.cat(
            (end_scores.to(dtype=torch.float32), end_scores_i.to(dtype=torch.float32).unsqueeze(0)), 0
        )
    return xy_preds, end_scores


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(
            maps,
            rois,
            widths_ceil,
            heights_ceil,
            widths,
            heights,
            offset_x,
            offset_y,
            torch.scalar_tensor(num_keypoints, dtype=torch.int64),
        )
        return xy_preds.permute(0, 2, 1), end_scores

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode="bicubic", align_corners=False
        )[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode="floor")
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints, device=roi_map.device), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    if H != W:
        raise ValueError(
            f"keypoint_logits height and width (last two elements of shape) should be equal. Instead got H = {H} and W = {W}"
        )
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.where(valid)[0]

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


def keypointrcnn_inference(x, boxes):
    # print(f'x:{x.shape}')
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)
    # print(f'x2:{x2}')

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores


def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]
    return im_mask


def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = box[2] - box[0] + one
    h = box[3] - box[1] + one
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0, unpaded_im_mask.to(dtype=torch.float32), zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0, concat_0, zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit._script_if_tracing
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(
            masks, boxes, torch.scalar_tensor(im_h, dtype=torch.int64), torch.scalar_tensor(im_w, dtype=torch.int64)
        )[:, None]
    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
            self,
            box_roi_pool,
            box_head,
            box_predictor,
            # Faster R-CNN training
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh,
            nms_thresh,
            detections_per_img,
            # Mask
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,
            keypoint_roi_pool=None,
            keypoint_head=None,
            keypoint_predictor=None,
            wirepoint_roi_pool=None,
            wirepoint_head=None,
            wirepoint_predictor=None,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.wirepoint_roi_pool = wirepoint_roi_pool
        self.wirepoint_head = wirepoint_head
        self.wirepoint_predictor = wirepoint_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def has_wirepoint(self):
        if self.wirepoint_roi_pool is None:
            print(f'wirepoint_roi_pool is None')
            return False
        if self.wirepoint_head is None:
            print(f'wirepoint_head is None')
            return False
        if self.wirepoint_predictor is None:
            print(f'wirepoint_roi_predictor is None')
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if self.has_mask():
            if not all(["masks" in t for t in targets]):
                raise ValueError("Every element of targets should have a masks key")

    def select_training_samples(
            self,
            proposals,  # type: List[Tensor]
            targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
            self,
            class_logits,  # type: Tensor
            box_regression,  # type: Tensor
            proposals,  # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
            self,
            features,  # type: Dict[str, Tensor]
            proposals,  # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
            targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            print('result append boxes!!!')
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.has_keypoint():

            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            # tmp = keypoint_features[0][0]
            # plt.imshow(tmp.detach().numpy())
            # print(f'keypoint_features from roi_pool:{keypoint_features.shape}')
            keypoint_features = self.keypoint_head(keypoint_features)

            # print(f'keypoint_features:{keypoint_features.shape}')
            tmp = keypoint_features[0][0]
            plt.imshow(tmp.detach().numpy())
            keypoint_logits = self.keypoint_predictor(keypoint_features)
            # print(f'keypoint_logits:{keypoint_logits.shape}')
            """
            接wirenet
            """

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        if self.has_wirepoint():
            # print(f'result:{result}')
            wirepoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                wirepoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    wirepoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            # print(f'proposals:{len(proposals)}')
            wirepoint_features = self.wirepoint_roi_pool(features, wirepoint_proposals, image_shapes)

            # tmp = keypoint_features[0][0]
            # plt.imshow(tmp.detach().numpy())
            # print(f'keypoint_features from roi_pool:{wirepoint_features.shape}')
            outputs, wirepoint_features = self.wirepoint_head(wirepoint_features)

            outputs = merge_features(outputs, wirepoint_proposals)
            wirepoint_features = merge_features(wirepoint_features, wirepoint_proposals)

            print(f'outpust:{outputs.shape}')

            wirepoint_logits = self.wirepoint_predictor(inputs=outputs, features=wirepoint_features, targets=targets)
            x, y, idx, jcs, n_batch, ps, n_out_line, n_out_junc = wirepoint_logits

            # print(f'keypoint_features:{wirepoint_features.shape}')
            if self.training:

                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                loss_weight = {'junc_map': 8.0, 'line_map': 0.5, 'junc_offset': 0.25, 'lpos': 1, 'lneg': 1}
                rcnn_loss_wirepoint = wirepoint_head_line_loss(targets, outputs, x, y, idx, loss_weight)

                loss_wirepoint = {"loss_wirepoint": rcnn_loss_wirepoint}

            else:
                pred = wirepoint_inference(x, idx, jcs, n_batch, ps, n_out_line, n_out_junc)
                result.append(pred)

                loss_wirepoint = {}

                # loss_weight = {'junc_map': 8.0, 'line_map': 0.5, 'junc_offset': 0.25, 'lpos': 1, 'lneg': 1}
                # rcnn_loss_wirepoint = wirepoint_head_line_loss(targets, outputs, x, y, idx, loss_weight)
                # loss_wirepoint = {"loss_wirepoint": rcnn_loss_wirepoint}

            # tmp = wirepoint_features[0][0]
            # plt.imshow(tmp.detach().numpy())
            # wirepoint_logits = self.wirepoint_predictor((outputs,wirepoint_features))
            # print(f'keypoint_logits:{wirepoint_logits.shape}')

            # loss_wirepoint = {}    lm
            # result=wirepoint_logits

            # result.append(pred)    lm
            losses.update(loss_wirepoint)
        # print(f"result{result}")
        # print(f"losses{losses}")

        return result, losses


# def merge_features(features, proposals):
#     # 假设 roi_pool_features 是你的输入张量，形状为 [600, 256, 128, 128]
#
#     # 使用 torch.split 按照每个图像的提议数量分割 features
#     proposals_count = sum([p.size(0) for p in proposals])
#     features_size = features.size(0)
#     # (f'proposals sum:{proposals_count},features batch:{features.size(0)}')
#     if proposals_count != features_size:
#         raise ValueError("The length of proposals must match the batch size of features.")
#
#     split_features = []
#     start_idx = 0
#     print(f"proposals:{proposals}")
#     for proposal in proposals:
#         # 提取当前图像的特征
#         current_features = features[start_idx:start_idx + proposal.size(0)]
#         # print(f'current_features:{current_features.shape}')
#         split_features.append(current_features)
#         start_idx += 1
#
#     features_imgs = []
#     for features_per_img in split_features:
#         features_per_img, _ = torch.max(features_per_img, dim=0, keepdim=True)
#         features_imgs.append(features_per_img)
#
#     merged_features = torch.cat(features_imgs, dim=0)
#     # print(f' merged_features:{merged_features.shape}')
#     return merged_features

def merge_features(features, proposals):
    print(f'features:{features.shape}')
    def diagnose_input(features, proposals):
        """诊断输入数据"""
        print("Input Diagnostics:")
        print(f"Features type: {type(features)}, shape: {features.shape}")
        print(f"Proposals type: {type(proposals)}, length: {len(proposals)}")
        for i, p in enumerate(proposals):
            print(f"Proposal {i} shape: {p.shape}")

    def validate_inputs(features, proposals):
        """验证输入的有效性"""
        if features is None or proposals is None:
            raise ValueError("Features or proposals cannot be None")

        proposals_count = sum([p.size(0) for p in proposals])
        features_size = features.size(0)

        if proposals_count != features_size:
            raise ValueError(
                f"Proposals count ({proposals_count}) must match features batch size ({features_size})"
            )

    def safe_max_reduction(features_per_img):
        """安全的最大值压缩"""
        if features_per_img.numel() == 0:
            return torch.zeros_like(features_per_img).unsqueeze(0)

        try:
            # 沿着第0维求最大值，保持维度
            max_features, _ = torch.max(features_per_img, dim=0, keepdim=True)
            return max_features
        except Exception as e:
            print(f"Max reduction error: {e}")
            return features_per_img.unsqueeze(0)

    try:
        # 诊断输入（可选）
        # diagnose_input(features, proposals)

        # 验证输入
        validate_inputs(features, proposals)

        # 分割特征
        split_features = []
        start_idx = 0

        for proposal in proposals:
            # 提取当前图像的特征
            current_features = features[start_idx:start_idx + proposal.size(0)]
            split_features.append(current_features)
            start_idx += proposal.size(0)

        # 每张图像特征压缩
        features_imgs = []
        for features_per_img in split_features:
            compressed_features = safe_max_reduction(features_per_img)
            features_imgs.append(compressed_features)

        # 合并特征
        merged_features = torch.cat(features_imgs, dim=0)

        return merged_features

    except Exception as e:
        print(f"Error in merge_features: {e}")
        # 返回原始特征或None
        return features

