from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.transforms import Resize as Scale


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        _cnn = models.regnet_x_400mf(weights='IMAGENET1K_V1')

        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4", 
                "trunk_output.block4": "c5",
            },
        )

        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        
        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out.items():
            print(f"Shape of {level_name} features: {feature_shape}")

        self.fpn_params = nn.ModuleDict()

        self.fpn_params['m3'] = nn.Conv2d(dummy_out['c3'].shape[1], out_channels, 
                                  kernel_size=1, stride=1, padding=0)
        self.fpn_params['m4'] = nn.Conv2d(dummy_out['c4'].shape[1], out_channels, 
                                  kernel_size=1, stride=1, padding=0)
        self.fpn_params['m5'] = nn.Conv2d(dummy_out['c5'].shape[1], out_channels, 
                                  kernel_size=1, stride=1, padding=0)

        self.fpn_params['p3'] = nn.Conv2d(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.fpn_params['p4'] = nn.Conv2d(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.fpn_params['p5'] = nn.Conv2d(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)

    @property
    def fpn_strides(self):
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}

        m5 = self.fpn_params['m5'](backbone_feats['c5'])
        m4 = self.fpn_params['m4'](backbone_feats['c4'])
        m3 = self.fpn_params['m3'](backbone_feats['c3'])
        m5_upsampled = F.interpolate(m5, size=(m4.shape[2], m4.shape[3]), mode='nearest')
        m4 = m4 + m5_upsampled
        
        m4_upsampled = F.interpolate(m4, size=(m3.shape[2], m3.shape[3]), mode='nearest')
        m3 = m3 + m4_upsampled
        
        fpn_feats['p5'] = self.fpn_params['p5'](m5)
        fpn_feats['p4'] = self.fpn_params['p4'](m4)
        fpn_feats['p3'] = self.fpn_params['p3'](m3)

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        H = feat_shape[2]
        W = feat_shape[3]
        x = torch.arange(0.5, W + 0.5, step=1, dtype=dtype, device=device) * level_stride
        y = torch.arange(0.5, H + 0.5, step=1, dtype=dtype, device=device) * level_stride
        (grid_x, grid_y) = torch.meshgrid(x, y, indexing='xy')
        grid_x = grid_x.unsqueeze(dim=-1)
        grid_y = grid_y.unsqueeze(dim=-1)
        location_coords[level_name] = torch.cat((grid_x, grid_y), dim=2).view(H*W, 2)
        
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = []
    x1, y1, x2, y2 = boxes[:, :4].unbind(dim=1)
    area = torch.mul(x2 - x1, y2 - y1)
    _, index = scores.sort(0)

    while index.numel() > 0:
        largest_idx = index[-1]
        keep.append(largest_idx)
        index = index[:-1]
        
        if index.size(0) == 0:
            break

        x1_inter = torch.index_select(x1, 0, index).clamp(min=x1[largest_idx])
        y1_inter = torch.index_select(y1, 0, index).clamp(min=y1[largest_idx])
        x2_inter = torch.index_select(x2, 0, index).clamp(max=x2[largest_idx])
        y2_inter = torch.index_select(y2, 0, index).clamp(max=y2[largest_idx])

        W_inter = (x2_inter - x1_inter).clamp(min=0.0)
        H_inter = (y2_inter - y1_inter).clamp(min=0.0)
        inter_area = W_inter * H_inter

        areas = torch.index_select(area, 0, index)
        union_area = (areas - inter_area) + area[largest_idx]

        IoU = inter_area / union_area
        index = index[IoU.le(iou_threshold)]

    keep = torch.tensor(keep, device=scores.device, dtype=torch.long)
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep