import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from a4_helper import *
from common import class_spec_nms, get_fpn_location_coords, nms
from torch import nn
from torch.nn import functional as F

TensorDict = Dict[str, torch.Tensor]


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")


class RPNPredictionNetwork(nn.Module):
    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        super().__init__()

        self.num_anchors = num_anchors
        
        stem_rpn = []
        prev_channels = in_channels
        for channels in stem_channels:
            stem_rpn.append(nn.Conv2d(prev_channels, channels, 3, padding=1))
            stem_rpn.append(nn.ReLU())
            prev_channels = channels
        
        for layer in stem_rpn:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        self.stem_rpn = nn.Sequential(*stem_rpn)

        self.pred_obj = nn.Conv2d(stem_channels[-1], num_anchors, 1)
        self.pred_box = nn.Conv2d(stem_channels[-1], num_anchors * 4, 1)
        
        nn.init.normal_(self.pred_obj.weight, mean=0, std=0.01)
        nn.init.normal_(self.pred_box.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_obj.bias, 0)
        nn.init.constant_(self.pred_box.bias, 0)

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        object_logits = {}
        boxreg_deltas = {}

        for level_name, features in feats_per_fpn_level.items():
            batch_size, channels, height, width = features.shape
            
            stem_features = self.stem_rpn(features)
            
            obj_logits = self.pred_obj(stem_features)
            box_deltas = self.pred_box(stem_features)
            
            obj_logits = obj_logits.view(batch_size, -1)
            
            box_deltas = box_deltas.view(batch_size, self.num_anchors, 4, height, width)
            box_deltas = box_deltas.permute(0, 3, 4, 1, 2).contiguous()
            box_deltas = box_deltas.view(batch_size, -1, 4)
            
            object_logits[level_name] = obj_logits
            boxreg_deltas[level_name] = box_deltas

        return [object_logits, boxreg_deltas]


@torch.no_grad()
def generate_fpn_anchors(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    stride_scale: int,
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
):
    anchors_per_fpn_level = {
        level_name: None for level_name, _ in locations_per_fpn_level.items()
    }

    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            area = (stride_scale * level_stride) ** 2
            width = torch.sqrt(torch.tensor(area / aspect_ratio))
            height = area / width
            
            half_width = width / 2
            half_height = height / 2
            
            x1 = locations[:, 0] - half_width
            y1 = locations[:, 1] - half_height
            x2 = locations[:, 0] + half_width
            y2 = locations[:, 1] + half_height
            
            anchor_box = torch.stack([x1, y1, x2, y2], dim=1)
            anchor_boxes.append(anchor_box)

        anchor_boxes = torch.stack(anchor_boxes)
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes

    return anchors_per_fpn_level


@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    x1_inter = torch.max(x1_1[:, None], x1_2[None, :])
    y1_inter = torch.max(y1_1[:, None], y1_2[None, :])
    x2_inter = torch.min(x2_1[:, None], x2_2[None, :])
    y2_inter = torch.min(y2_1[:, None], y2_2[None, :])
    
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    intersection = inter_width * inter_height
    
    union = area1[:, None] + area2[None, :] - intersection
    
    iou_matrix = intersection / torch.clamp(union, min=1e-8)
    
    return iou_matrix


@torch.no_grad()
def rcnn_match_anchors_to_gt(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, float],
) -> TensorDict:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]

    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)

    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])

    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]

    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1

    neutral_idxs = (match_quality > iou_thresholds[0]) & (
        match_quality < iou_thresholds[1]
    )
    matched_gt_boxes[neutral_idxs, :] = -1e8
    return matched_gt_boxes


def rcnn_get_deltas_from_anchors(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    background_mask = (gt_boxes[:, 0] <= -1) | (gt_boxes[:, 0] <= -1e7)
    
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
    anchor_cy = anchors[:, 1] + 0.5 * anchor_heights
    
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_heights
    
    dx = (gt_cx - anchor_cx) / anchor_widths
    dy = (gt_cy - anchor_cy) / anchor_heights
    dw = torch.log(gt_widths / anchor_widths)
    dh = torch.log(gt_heights / anchor_heights)
    
    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    
    deltas[background_mask] = -1e8
    
    return deltas


def rcnn_apply_deltas_to_anchors(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    scale_clamp = math.log(224 / 8)
    deltas[:, 2] = torch.clamp(deltas[:, 2], max=scale_clamp)
    deltas[:, 3] = torch.clamp(deltas[:, 3], max=scale_clamp)

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
    anchor_cy = anchors[:, 1] + 0.5 * anchor_heights
    
    pred_cx = deltas[:, 0] * anchor_widths + anchor_cx
    pred_cy = deltas[:, 1] * anchor_heights + anchor_cy
    pred_w = torch.exp(deltas[:, 2]) * anchor_widths
    pred_h = torch.exp(deltas[:, 3]) * anchor_heights
    
    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h
    
    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return output_boxes


@torch.no_grad()
def sample_rpn_training(
    gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float
):
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)

    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg

    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]

    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return fg_idx, bg_idx


@torch.no_grad()
def mix_gt_with_proposals(
    proposals_per_fpn_level: Dict[str, List[torch.Tensor]], gt_boxes: torch.Tensor
):
    for _idx, _gtb in enumerate(gt_boxes):
        _gtb = _gtb[_gtb[:, 4] != -1]
        if len(_gtb) == 0:
            continue

        _gt_area = (_gtb[:, 2] - _gtb[:, 0]) * (_gtb[:, 3] - _gtb[:, 1])
        level_assn = torch.floor(5 + torch.log2(torch.sqrt(_gt_area) / 224))
        level_assn = torch.clamp(level_assn, min=3, max=5).to(torch.int64)

        for level_name, _props in proposals_per_fpn_level.items():
            _prop = _props[_idx]

            _gt_boxes_fpn_subset = _gtb[level_assn == int(level_name[1])]
            if len(_gt_boxes_fpn_subset) > 0:
                proposals_per_fpn_level[level_name][_idx] = torch.cat(
                    [_prop, _gt_boxes_fpn_subset[:, :4]],
                    dim=0,
                )

    return proposals_per_fpn_level


class RPN(nn.Module):
    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        num_images = feats_per_fpn_level["p3"].shape[0]

        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)
        
        locations_per_fpn_level = get_fpn_location_coords(
            feats_per_fpn_level, strides_per_fpn_level
        )
        anchors_per_fpn_level = generate_fpn_anchors(
            locations_per_fpn_level,
            strides_per_fpn_level,
            self.anchor_stride_scale,
            self.anchor_aspect_ratios
        )

        output_dict = {}

        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        if not self.training:
            return output_dict

        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        matched_gt_boxes = []
        for gt_boxes_per_image in gt_boxes:
            matched_gt_boxes.append(
                rcnn_match_anchors_to_gt(
                    anchor_boxes, gt_boxes_per_image, self.anchor_iou_thresholds
                )
            )

        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            fg_idx, bg_idx = sample_rpn_training(
                matched_gt_boxes, self.batch_size_per_image, 0.5
            )
            sampled_idx = torch.cat([fg_idx, bg_idx])
            
            sampled_gt_boxes = matched_gt_boxes[sampled_idx]
            sampled_obj_logits = pred_obj_logits[sampled_idx]
            sampled_boxreg_deltas = pred_boxreg_deltas[sampled_idx]
            sampled_anchors = anchor_boxes[sampled_idx]
            
            gt_deltas = rcnn_get_deltas_from_anchors(sampled_anchors, sampled_gt_boxes)
            
            gt_objectness = (sampled_gt_boxes[:, 4] >= 0).float()
            loss_obj = F.binary_cross_entropy_with_logits(
                sampled_obj_logits, gt_objectness, reduction='none'
            )
            
            foreground_mask = (sampled_gt_boxes[:, 4] >= 0)
            loss_box = F.smooth_l1_loss(
                sampled_boxreg_deltas, gt_deltas, reduction='none'
            ).sum(dim=1)
            loss_box = loss_box * foreground_mask.float()

            total_batch_size = self.batch_size_per_image * num_images
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    @torch.no_grad()
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],
    ):
        proposals_all_levels = {
            level_name: None for level_name, _ in anchors_per_fpn_level.items()
        }
        for level_name in anchors_per_fpn_level.keys():
            level_anchors = anchors_per_fpn_level[level_name]

            level_obj_logits = pred_obj_logits[level_name]
            level_boxreg_deltas = pred_boxreg_deltas[level_name]

            level_proposals_per_image = []
            for _batch_idx in range(level_obj_logits.shape[0]):
                img_obj_logits = level_obj_logits[_batch_idx]
                img_boxreg_deltas = level_boxreg_deltas[_batch_idx]
                
                proposal_boxes = rcnn_apply_deltas_to_anchors(img_boxreg_deltas, level_anchors)
                
                proposal_boxes[:, [0, 2]] = torch.clamp(proposal_boxes[:, [0, 2]], 0, image_size[0])
                proposal_boxes[:, [1, 3]] = torch.clamp(proposal_boxes[:, [1, 3]], 0, image_size[1])
                
                obj_scores = torch.sigmoid(img_obj_logits)
                _, top_indices = torch.topk(obj_scores, min(self.pre_nms_topk, len(obj_scores)))
                
                top_boxes = proposal_boxes[top_indices]
                top_scores = obj_scores[top_indices]
                
                keep_indices = torchvision.ops.nms(top_boxes, top_scores, self.nms_thresh)
                keep_indices = keep_indices[:self.post_nms_topk]
                
                final_proposals = top_boxes[keep_indices]
                level_proposals_per_image.append(final_proposals)

            proposals_all_levels[level_name] = level_proposals_per_image

        return proposals_all_levels

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)


class FasterRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        stem_channels: List[int],
        num_classes: int,
        batch_size_per_image: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image

        cls_pred = []
        in_channels = backbone.fpn_channels
        prev_channels = in_channels
        
        for channels in stem_channels:
            cls_pred.append(nn.Conv2d(prev_channels, channels, 3, padding=1))
            cls_pred.append(nn.ReLU())
            prev_channels = channels
        
        for layer in cls_pred:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        cls_pred.append(nn.Flatten())
        linear_input_size = stem_channels[-1] * roi_size[0] * roi_size[1]
        cls_pred.append(nn.Linear(linear_input_size, num_classes + 1))
        
        nn.init.normal_(cls_pred[-1].weight, mean=0, std=0.01)
        nn.init.constant_(cls_pred[-1].bias, 0)

        self.cls_pred = nn.Sequential(*cls_pred)

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(
            feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes
        )
        proposals_per_fpn_level = output_dict["proposals"]

        if self.training:
            proposals_per_fpn_level = mix_gt_with_proposals(
                proposals_per_fpn_level, gt_boxes
            )

        num_images = feats_per_fpn_level["p3"].shape[0]

        roi_feats_per_fpn_level = {
            level_name: None for level_name in feats_per_fpn_level.keys()
        }
        for level_name in feats_per_fpn_level.keys():
            level_feats = feats_per_fpn_level[level_name]
            level_props = output_dict["proposals"][level_name]
            level_stride = self.backbone.fpn_strides[level_name]

            proposals_with_batch_idx = []
            for batch_idx, props in enumerate(level_props):
                if len(props) > 0:
                    batch_indices = torch.full((len(props), 1), batch_idx, 
                                             dtype=props.dtype, device=props.device)
                    props_with_idx = torch.cat([batch_indices, props], dim=1)
                    proposals_with_batch_idx.append(props_with_idx)
            
            if proposals_with_batch_idx:
                all_proposals = torch.cat(proposals_with_batch_idx, dim=0)
                roi_feats = torchvision.ops.roi_align(
                    level_feats,
                    all_proposals,
                    output_size=self.roi_size,
                    spatial_scale=1.0 / level_stride,
                    aligned=True
                )
            else:
                roi_feats = torch.empty(0, level_feats.shape[1], self.roi_size[0], self.roi_size[1],
                                      device=level_feats.device, dtype=level_feats.dtype)

            roi_feats_per_fpn_level[level_name] = roi_feats

        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)

        pred_cls_logits = self.cls_pred(roi_feats)

        if not self.training:
            return self.inference(
                images,
                proposals_per_fpn_level,
                pred_cls_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )

        matched_gt_boxes = []
        for _idx in range(len(gt_boxes)):
            proposals_per_fpn_level_per_image = {
                level_name: prop[_idx]
                for level_name, prop in output_dict["proposals"].items()
            }
            proposals_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            gt_boxes_per_image = gt_boxes[_idx]
            matched_gt_boxes.append(
                rcnn_match_anchors_to_gt(
                    proposals_per_image, gt_boxes_per_image, (0.5, 0.5)
                )
            )

        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

        fg_idx, bg_idx = sample_rpn_training(
            matched_gt_boxes, self.batch_size_per_image, 0.25
        )
        sampled_idx = torch.cat([fg_idx, bg_idx])
        
        sampled_gt_boxes = matched_gt_boxes[sampled_idx]
        sampled_cls_logits = pred_cls_logits[sampled_idx]
        
        gt_classes = sampled_gt_boxes[:, 4].long() + 1
        gt_classes[sampled_gt_boxes[:, 4] == -1] = 0
        
        loss_cls = F.cross_entropy(sampled_cls_logits, gt_classes)

        return {
            "loss_rpn_obj": output_dict["loss_rpn_obj"],
            "loss_rpn_box": output_dict["loss_rpn_box"],
            "loss_cls": loss_cls,
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        proposals: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        test_score_thresh: float,
        test_nms_thresh: float,
    ):
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)

        pred_probs = F.softmax(pred_cls_logits, dim=1)
        pred_scores, pred_classes = pred_probs.max(dim=1)
        
        keep_mask = (pred_scores > test_score_thresh) & (pred_classes > 0)
        pred_scores = pred_scores[keep_mask]
        pred_classes = pred_classes[keep_mask] - 1
        pred_boxes = pred_boxes[keep_mask]

        keep = class_spec_nms(
            pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh
        )
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_classes, pred_scores