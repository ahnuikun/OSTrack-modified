"""Read-only paired diagnostics for VDRM same-class Copy-Paste."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from lib.train.data.vdrm_augmentation import (
    apply_same_class_distractor_copy_paste,
)
from lib.utils.heapmap_utils import generate_heatmap


@torch.no_grad()
def create_paired_copy_pastes(
    clean_image: torch.Tensor,
    target_box: torch.Tensor,
    distractor_image: torch.Tensor,
    distractor_box: torch.Tensor,
    invalid_mask: torch.Tensor,
    min_scale: float,
    max_scale: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    bool,
    bool,
    torch.Tensor,
    torch.Tensor,
]:
    """Create random and nearest variants from exactly the same RNG state.

    The random variant consumes the first valid candidate, matching V3. The
    RNG state is then restored before the nearest variant, so it sees the same
    sampled scale and candidate prefix before evaluating all 24 candidates.
    """
    if clean_image.device.type != "cpu":
        raise ValueError("paired Copy-Paste preparation must run on CPU")

    rng_state = torch.get_rng_state()
    random_image, random_applied, random_box = (
        apply_same_class_distractor_copy_paste(
            clean_image,
            target_box,
            distractor_image,
            distractor_box,
            min_scale=min_scale,
            max_scale=max_scale,
            invalid_mask=invalid_mask,
            placement_mode="random",
        )
    )
    torch.set_rng_state(rng_state)
    nearest_image, nearest_applied, nearest_box = (
        apply_same_class_distractor_copy_paste(
            clean_image,
            target_box,
            distractor_image,
            distractor_box,
            min_scale=min_scale,
            max_scale=max_scale,
            invalid_mask=invalid_mask,
            placement_mode="nearest",
        )
    )
    return (
        random_image,
        nearest_image,
        random_applied,
        nearest_applied,
        random_box,
        nearest_box,
    )


def normalized_center_distance(
    first_box: torch.Tensor,
    second_box: torch.Tensor,
) -> torch.Tensor:
    """Return the size-normalized squared center distance for xywh boxes."""
    first_box = first_box.reshape(-1, 4)
    second_box = second_box.reshape(-1, 4)
    first_center = first_box[:, :2] + 0.5 * first_box[:, 2:]
    second_center = second_box[:, :2] + 0.5 * second_box[:, 2:]
    scale = (first_box[:, 2:] + second_box[:, 2:]).clamp_min(1e-12)
    return (((first_center - second_center) / scale).square()).sum(dim=1)


def _xywh_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.reshape(-1, 4)
    second = second.reshape(-1, 4)
    first_end = first[:, :2] + first[:, 2:]
    second_end = second[:, :2] + second[:, 2:]
    intersection_size = (
        torch.minimum(first_end, second_end)
        - torch.maximum(first[:, :2], second[:, :2])
    ).clamp_min(0.0)
    intersection = intersection_size.prod(dim=1)
    union = (
        first[:, 2:].prod(dim=1)
        + second[:, 2:].prod(dim=1)
        - intersection
    )
    return intersection / union.clamp_min(1e-12)


def _distractor_mask(
    boxes: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    """Map normalized boxes to response cells with the V5 tiny-box fallback."""
    batch_size, height, width = negative_mask.shape
    boxes = boxes.to(device=negative_mask.device).reshape(-1, 4)
    if boxes.shape[0] != batch_size:
        raise ValueError("distractor box batch size does not match responses")

    x0 = boxes[:, 0].clamp(0.0, 1.0)
    y0 = boxes[:, 1].clamp(0.0, 1.0)
    x1 = (boxes[:, 0] + boxes[:, 2]).clamp(0.0, 1.0)
    y1 = (boxes[:, 1] + boxes[:, 3]).clamp(0.0, 1.0)
    valid = (x1 > x0) & (y1 > y0) & negative_mask.flatten(1).any(dim=1)

    cell_x = (
        torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
    ) / width
    cell_y = (
        torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
    ) / height
    box_mask = (
        (cell_x[None, None, :] >= x0[:, None, None])
        & (cell_x[None, None, :] < x1[:, None, None])
        & (cell_y[None, :, None] >= y0[:, None, None])
        & (cell_y[None, :, None] < y1[:, None, None])
    )
    mask = box_mask & negative_mask
    missing = valid & ~mask.flatten(1).any(dim=1)
    if missing.any():
        center_x = ((x0 + x1) * 0.5)[:, None, None]
        center_y = ((y0 + y1) * 0.5)[:, None, None]
        distance = (
            (cell_x[None, None, :] - center_x).square()
            + (cell_y[None, :, None] - center_y).square()
        ).masked_fill(~negative_mask, torch.inf)
        nearest_index = distance.flatten(1).argmin(dim=1, keepdim=True)
        fallback = torch.zeros_like(negative_mask.flatten(1))
        fallback.scatter_(1, nearest_index, True)
        mask |= missing[:, None, None] & fallback.view(
            batch_size, height, width
        )
    return mask


@torch.no_grad()
def compute_condition_metrics(
    model_output: Dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    search_size: int,
    stride: int,
    distractor_boxes: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Extract per-sample response and box metrics from one model forward."""
    score_logits = model_output["score_logits"].detach().squeeze(1)
    score_map = model_output["score_map"].detach().squeeze(1)
    if score_logits.ndim != 3 or score_map.shape != score_logits.shape:
        raise ValueError("score logits/map must have matching [B,H,W] shapes")

    target_boxes = target_boxes.to(
        device=score_logits.device, dtype=score_logits.dtype
    ).reshape(-1, 4)
    gaussian = generate_heatmap(
        [target_boxes], patch_size=search_size, stride=stride
    )[0].to(device=score_logits.device, dtype=score_logits.dtype)
    positive_index = gaussian.flatten(1).argmax(dim=1, keepdim=True)
    negative_mask = gaussian <= 0.0
    if not negative_mask.flatten(1).any(dim=1).all():
        raise RuntimeError("one or more target heatmaps have no background")

    masked_logits = score_logits.masked_fill(~negative_mask, -torch.inf)
    masked_scores = score_map.masked_fill(~negative_mask, -torch.inf)
    global_negative_logit, global_negative_index = (
        masked_logits.flatten(1).max(dim=1)
    )
    global_negative_score = masked_scores.flatten(1).amax(dim=1)
    target_logit = score_logits.flatten(1).gather(
        1, positive_index
    ).squeeze(1)
    target_score = score_map.flatten(1).gather(
        1, positive_index
    ).squeeze(1)

    predicted_cxcywh = model_output["pred_boxes"].detach().mean(dim=1)
    predicted_xywh = torch.cat(
        (
            predicted_cxcywh[:, :2] - 0.5 * predicted_cxcywh[:, 2:],
            predicted_cxcywh[:, 2:],
        ),
        dim=1,
    )
    target_center = target_boxes[:, :2] + 0.5 * target_boxes[:, 2:]
    predicted_center = predicted_xywh[:, :2] + 0.5 * predicted_xywh[:, 2:]
    center_error = (
        (predicted_center - target_center).square().sum(dim=1).sqrt()
        / target_boxes[:, 2:].prod(dim=1).sqrt().clamp_min(1e-12)
    )

    visual_reliability = model_output.get("visual_reliability")
    if visual_reliability is None:
        visual_reliability = target_logit.new_full(target_logit.shape, torch.nan)
    else:
        visual_reliability = visual_reliability.detach().reshape(
            target_logit.shape[0], -1
        ).mean(dim=1)

    metrics = {
        "target_logit": target_logit,
        "target_score": target_score,
        "global_negative_logit": global_negative_logit,
        "global_negative_score": global_negative_score,
        "rank_margin": target_logit - global_negative_logit,
        "pred_iou": _xywh_iou(predicted_xywh, target_boxes),
        "pred_center_error": center_error,
        "visual_reliability": visual_reliability,
    }

    if distractor_boxes is not None:
        distractor_boxes = distractor_boxes.to(
            device=score_logits.device, dtype=score_logits.dtype
        ).reshape(-1, 4)
        paste_mask = _distractor_mask(distractor_boxes, negative_mask)
        paste_logit = score_logits.masked_fill(
            ~paste_mask, -torch.inf
        ).flatten(1).amax(dim=1)
        paste_score = score_map.masked_fill(
            ~paste_mask, -torch.inf
        ).flatten(1).amax(dim=1)
        global_inside_paste = paste_mask.flatten(1).gather(
            1, global_negative_index[:, None]
        ).squeeze(1)
        metrics.update({
            "paste_logit": paste_logit,
            "paste_score": paste_score,
            "paste_global_gap": global_negative_logit - paste_logit,
            "paste_hard_hit": global_inside_paste.to(score_logits.dtype),
        })
    return metrics


def summarize_pair_rows(rows: List[dict]) -> dict:
    """Summarize explicit paired metrics without introducing thresholds."""
    if not rows:
        raise ValueError("cannot summarize an empty paired diagnostic")

    def mean(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        values = values[torch.isfinite(values)]
        return values.mean().item() if values.numel() else float("nan")

    condition_metrics = (
        "target_logit",
        "target_score",
        "global_negative_logit",
        "global_negative_score",
        "rank_margin",
        "pred_iou",
        "pred_center_error",
        "visual_reliability",
    )
    summary = {
        "sample_count": len(rows),
        "conditions": {},
        "placement": {
            "random_distance": mean("random_distance"),
            "near_distance": mean("near_distance"),
            "random_paste_logit": mean("random_paste_logit"),
            "near_paste_logit": mean("near_paste_logit"),
            "random_global_gap": mean("random_paste_global_gap"),
            "near_global_gap": mean("near_paste_global_gap"),
            "random_hard_hit_rate": mean("random_paste_hard_hit"),
            "near_hard_hit_rate": mean("near_paste_hard_hit"),
        },
    }
    for condition in ("clean", "random", "near"):
        summary["conditions"][condition] = {
            metric: mean(f"{condition}_{metric}")
            for metric in condition_metrics
        }

    comparisons = {}
    for first, second, name in (
        ("random", "clean", "random_minus_clean"),
        ("near", "clean", "near_minus_clean"),
        ("near", "random", "near_minus_random"),
    ):
        comparisons[name] = {
            metric: mean(f"{first}_{metric}") - mean(f"{second}_{metric}")
            for metric in condition_metrics
        }
    comparisons["near_minus_random"].update({
        "paste_logit": (
            mean("near_paste_logit") - mean("random_paste_logit")
        ),
        "paste_global_gap": (
            mean("near_paste_global_gap")
            - mean("random_paste_global_gap")
        ),
        "paste_hard_hit_rate": (
            mean("near_paste_hard_hit") - mean("random_paste_hard_hit")
        ),
    })
    summary["comparisons"] = comparisons
    return summary
