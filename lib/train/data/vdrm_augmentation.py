"""Training-only structured target occlusion for VDRM."""

from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def apply_structured_target_occlusion(
    images: torch.Tensor,
    target_boxes: torch.Tensor,
    probability: float = 0.5,
    min_area_ratio: float = 0.2,
    max_area_ratio: float = 0.5,
    part_grid: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask one contiguous rectangle inside each selected search target.

    Args:
        images: normalized search images, ``[B, C, H, W]``.
        target_boxes: normalized ``xywh`` target boxes, ``[B, 4]``.
        probability: per-sample augmentation probability.
        min_area_ratio: minimum occluded fraction of the target box.
        max_area_ratio: maximum occluded fraction of the target box.
        part_grid: target-local visibility grid size.

    Returns:
        occluded_images: a cloned tensor with selected rectangles filled by 0.
        part_visibility: remaining visible fraction for every target part,
            shape ``[B, part_grid ** 2]``.
        applied: samples with a valid synthetic occlusion, shape ``[B]``.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B, C, H, W], got {images.shape}")
    if target_boxes.shape != (images.shape[0], 4):
        raise ValueError(
            f"target_boxes must have shape [{images.shape[0]}, 4], "
            f"got {target_boxes.shape}"
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if not 0.0 < min_area_ratio <= max_area_ratio < 1.0:
        raise ValueError(
            "area ratios must satisfy 0 < min <= max < 1, got "
            f"{min_area_ratio}, {max_area_ratio}"
        )
    if part_grid < 1:
        raise ValueError(f"part_grid must be positive, got {part_grid}")

    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype

    boxes = target_boxes.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    box_x0 = boxes[:, 0] * width
    box_y0 = boxes[:, 1] * height
    box_x1 = ((boxes[:, 0] + boxes[:, 2]) * width).clamp(0.0, float(width))
    box_y1 = ((boxes[:, 1] + boxes[:, 3]) * height).clamp(0.0, float(height))
    box_w = (box_x1 - box_x0).clamp_min(0.0)
    box_h = (box_y1 - box_y0).clamp_min(0.0)

    applied = torch.rand(batch_size, device=device) < probability
    applied &= (box_w >= 2.0) & (box_h >= 2.0)

    area_ratio = torch.empty(batch_size, device=device).uniform_(
        min_area_ratio, max_area_ratio
    )
    log_aspect = torch.empty(batch_size, device=device).uniform_(
        -0.5, 0.5
    )
    aspect = log_aspect.exp()
    occ_w_ratio = torch.sqrt(area_ratio * aspect).clamp(max=0.95)
    occ_h_ratio = torch.sqrt(area_ratio / aspect).clamp(max=0.95)
    occ_w = box_w * occ_w_ratio
    occ_h = box_h * occ_h_ratio

    occ_x0 = box_x0 + torch.rand(batch_size, device=device) * (box_w - occ_w)
    occ_y0 = box_y0 + torch.rand(batch_size, device=device) * (box_h - occ_h)
    occ_x1 = occ_x0 + occ_w
    occ_y1 = occ_y0 + occ_h

    pixel_x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    pixel_y = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    mask_x = (
        (pixel_x[None, :] >= occ_x0[:, None])
        & (pixel_x[None, :] < occ_x1[:, None])
    )
    mask_y = (
        (pixel_y[None, :] >= occ_y0[:, None])
        & (pixel_y[None, :] < occ_y1[:, None])
    )
    occlusion_mask = (
        mask_y[:, :, None] & mask_x[:, None, :] & applied[:, None, None]
    )

    occluded_images = images.clone()
    occluded_images.masked_fill_(occlusion_mask[:, None], 0.0)

    visibility_parts = []
    for row in range(part_grid):
        part_y0 = box_y0 + box_h * (row / part_grid)
        part_y1 = box_y0 + box_h * ((row + 1) / part_grid)
        for col in range(part_grid):
            part_x0 = box_x0 + box_w * (col / part_grid)
            part_x1 = box_x0 + box_w * ((col + 1) / part_grid)

            overlap_w = (
                torch.minimum(part_x1, occ_x1)
                - torch.maximum(part_x0, occ_x0)
            ).clamp_min(0.0)
            overlap_h = (
                torch.minimum(part_y1, occ_y1)
                - torch.maximum(part_y0, occ_y0)
            ).clamp_min(0.0)
            part_area = ((part_x1 - part_x0) * (part_y1 - part_y0)).clamp_min(
                1.0
            )
            visibility = 1.0 - (overlap_w * overlap_h / part_area)
            visibility = torch.where(
                applied, visibility, torch.ones_like(visibility)
            )
            visibility_parts.append(visibility)

    part_visibility = torch.stack(visibility_parts, dim=-1)
    return (
        occluded_images,
        part_visibility.to(dtype=dtype),
        applied,
    )
