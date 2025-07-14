import math
from typing import Dict, List, Optional

import torch
from a4_helper import *
from common import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss

TensorDict = Dict[str, torch.Tensor]


def hello_one_stage_detector():
    print("Hello from one_stage_detector.py!")


class FCOSPredictionNetwork(nn.Module):
    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        super().__init__()

        stem_cls = []
        stem_box = []
        
        # Create stem layers for classification
        prev_channels = in_channels
        for channels in stem_channels:
            stem_cls.append(nn.Conv2d(prev_channels, channels, 3, padding=1))
            stem_cls.append(nn.ReLU())
            prev_channels = channels
        
        # Create stem layers for box regression
        prev_channels = in_channels
        for channels in stem_channels:
            stem_box.append(nn.Conv2d(prev_channels, channels, 3, padding=1))
            stem_box.append(nn.ReLU())
            prev_channels = channels
        
        # Initialize weights
        for layer in stem_cls:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        for layer in stem_box:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Prediction layers
        self.pred_cls = nn.Conv2d(stem_channels[-1], num_classes, 3, padding=1)
        self.pred_box = nn.Conv2d(stem_channels[-1], 4, 3, padding=1)
        self.pred_ctr = nn.Conv2d(stem_channels[-1], 1, 3, padding=1)
        
        # Initialize prediction layers
        nn.init.normal_(self.pred_cls.weight, mean=0, std=0.01)
        nn.init.normal_(self.pred_box.weight, mean=0, std=0.01)
        nn.init.normal_(self.pred_ctr.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_cls.bias, 0)
        nn.init.constant_(self.pred_box.bias, 0)
        nn.init.constant_(self.pred_ctr.bias, 0)

        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for level_name, features in feats_per_fpn_level.items():
            batch_size, channels, height, width = features.shape
            
            # Pass through stems
            cls_features = self.stem_cls(features)
            box_features = self.stem_box(features)
            
            # Get predictions
            cls_logits = self.pred_cls(cls_features)
            box_deltas = self.pred_box(box_features)
            ctr_logits = self.pred_ctr(box_features)
            
            # Reshape to (batch_size, H*W, num_outputs)
            cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)
            box_deltas = box_deltas.permute(0, 2, 3, 1).reshape(batch_size, height * width, 4)
            ctr_logits = ctr_logits.permute(0, 2, 3, 1).reshape(batch_size, height * width, 1)
            
            class_logits[level_name] = cls_logits
            boxreg_deltas[level_name] = box_deltas
            centerness_logits[level_name] = ctr_logits

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    for level_name, centers in locations_per_fpn_level.items():
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        match_matrix = pairwise_dist.min(dim=2).values > 0

        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    # Handle background boxes
    background_mask = (gt_boxes[:, 0] == -1) & (gt_boxes[:, 1] == -1) & (gt_boxes[:, 2] == -1) & (gt_boxes[:, 3] == -1)
    
    # Calculate deltas: left, top, right, bottom
    left = locations[:, 0] - gt_boxes[:, 0]
    top = locations[:, 1] - gt_boxes[:, 1]
    right = gt_boxes[:, 2] - locations[:, 0]
    bottom = gt_boxes[:, 3] - locations[:, 1]
    
    # Stack deltas
    deltas = torch.stack([left, top, right, bottom], dim=1)
    
    # Normalize by stride
    deltas = deltas / stride
    
    # Set background boxes to -1
    deltas[background_mask] = -1
    
    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    # Un-normalize deltas
    deltas = deltas * stride
    
    # Clip negative deltas to zero
    deltas = torch.clamp(deltas, min=0)
    
    # Apply deltas to locations
    x1 = locations[:, 0] - deltas[:, 0]  # left
    y1 = locations[:, 1] - deltas[:, 1]  # top
    x2 = locations[:, 0] + deltas[:, 2]  # right
    y2 = locations[:, 1] + deltas[:, 3]  # bottom
    
    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    # Handle background boxes
    background_mask = (deltas == -1).all(dim=1)
    
    # Calculate centerness for non-background boxes
    left_right = torch.min(deltas[:, 0], deltas[:, 2]) / torch.max(deltas[:, 0], deltas[:, 2])
    top_bottom = torch.min(deltas[:, 1], deltas[:, 3]) / torch.max(deltas[:, 1], deltas[:, 3])
    
    centerness = torch.sqrt(left_right * top_bottom)
    
    # Set background boxes to -1
    centerness[background_mask] = -1
    
    return centerness


class FCOS(nn.Module):
    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes, fpn_channels, stem_channels)

        self._normalizer = 150

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        # Get FPN features
        feats_per_fpn_level = self.backbone(images)
        
        # Get predictions
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(feats_per_fpn_level)

        # Get locations
        locations_per_fpn_level = get_fpn_location_coords(
            feats_per_fpn_level, self.backbone.fpn_strides
        )

        if not self.training:
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )

        # Training mode: compute losses
        matched_gt_boxes = []
        for gt_boxes_per_image in gt_boxes:
            matched_gt_boxes.append(
                fcos_match_locations_to_gt(
                    locations_per_fpn_level,
                    self.backbone.fpn_strides,
                    gt_boxes_per_image
                )
            )

        matched_gt_deltas = []
        for level_name in locations_per_fpn_level.keys():
            level_deltas = []
            for i, matched_boxes in enumerate(matched_gt_boxes):
                level_deltas.append(
                    fcos_get_deltas_from_locations(
                        locations_per_fpn_level[level_name],
                        matched_boxes[level_name],
                        self.backbone.fpn_strides[level_name]
                    )
                )
            matched_gt_deltas.append({level_name: torch.stack(level_deltas)})
        
        # Reorganize matched_gt_deltas
        matched_gt_deltas_dict = {}
        for level_name in locations_per_fpn_level.keys():
            matched_gt_deltas_dict[level_name] = torch.stack([
                fcos_get_deltas_from_locations(
                    locations_per_fpn_level[level_name],
                    matched_gt_boxes[i][level_name],
                    self.backbone.fpn_strides[level_name]
                ) for i in range(len(matched_gt_boxes))
            ])
        matched_gt_deltas = matched_gt_deltas_dict

        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        # Compute losses
        # Classification loss
        gt_classes = matched_gt_boxes[:, :, 4].long()
        foreground_mask = gt_classes != -1
        gt_classes_one_hot = F.one_hot(gt_classes.clamp(min=0), self.num_classes).float()
        loss_cls = sigmoid_focal_loss(
            pred_cls_logits, gt_classes_one_hot, reduction='none'
        ).sum(dim=2)
        loss_cls = loss_cls * foreground_mask.float()

        # Box regression loss
        loss_box = F.l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction='none').sum(dim=2)
        loss_box = loss_box * foreground_mask.float()

        # Centerness loss
        gt_centerness = fcos_make_centerness_targets(matched_gt_deltas).unsqueeze(2)
        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits, gt_centerness, reduction='none'
        ).squeeze(2)
        loss_ctr = loss_ctr * foreground_mask.float()

        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            # Compute geometric mean of class probability and centerness
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            
            # Step 1: Get most confident class and score
            level_pred_scores, level_pred_classes = level_pred_scores.max(dim=1)
            
            # Step 2: Filter by threshold
            keep_mask = level_pred_scores > test_score_thresh
            level_pred_scores = level_pred_scores[keep_mask]
            level_pred_classes = level_pred_classes[keep_mask]
            level_locations = level_locations[keep_mask]
            level_deltas = level_deltas[keep_mask]
            
            # Step 3: Apply deltas to get boxes
            level_pred_boxes = fcos_apply_deltas_to_locations(
                level_deltas, level_locations, self.backbone.fpn_strides[level_name]
            )
            
            # Step 4: Clip boxes to image boundaries
            height, width = images.shape[2], images.shape[3]
            level_pred_boxes[:, [0, 2]] = torch.clamp(level_pred_boxes[:, [0, 2]], 0, width)
            level_pred_boxes[:, [1, 3]] = torch.clamp(level_pred_boxes[:, [1, 3]], 0, height)

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )