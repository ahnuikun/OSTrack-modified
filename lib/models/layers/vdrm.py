"""Visibility-driven representation module for OSTrack.

The first implementation intentionally keeps the design small:

* the template target is split into a fixed 2 x 2 grid;
* each grid cell is represented by masked mean pooling;
* cosine matching estimates whether each part is visible in the search;
* matched, reliable prototypes are added to search tokens through one
  zero-initialized residual scalar.

No temporal state or motion information is used here.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class VisibilityDrivenRepresentationModule(nn.Module):
    """Lightweight fixed-part prototype matching and residual refinement."""

    def __init__(
        self,
        num_parts: int = 4,
        topk: int = 4,
        initial_match_scale: float = 5.0,
        initial_match_bias: float = -2.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        part_grid = int(math.sqrt(num_parts))
        if part_grid * part_grid != num_parts:
            raise ValueError(f"num_parts must be a square number, got {num_parts}")
        if topk < 1:
            raise ValueError(f"topk must be positive, got {topk}")

        self.num_parts = num_parts
        self.part_grid = part_grid
        self.topk = topk
        self.eps = eps

        # A shared monotonic calibration is enough for the first version.
        # softplus(log_match_scale) keeps higher similarity mapped to higher
        # reliability without adding an MLP or another gating branch.
        initial_scale_tensor = torch.tensor(float(initial_match_scale))
        self.log_match_scale = nn.Parameter(
            torch.log(torch.expm1(initial_scale_tensor))
        )
        self.match_bias = nn.Parameter(torch.tensor(float(initial_match_bias)))

        # Zero initialization preserves the original OSTrack forward path.
        self.alpha = nn.Parameter(torch.zeros(()))

    def _default_template_bbox(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a conservative centered template box for API fallback."""
        return torch.tensor(
            [0.25, 0.25, 0.5, 0.5], device=device, dtype=dtype
        ).expand(batch_size, -1)

    def _build_part_masks(
        self,
        template_bbox: torch.Tensor,
        template_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create fixed target-local grid masks on the template token grid.

        Args:
            template_bbox: normalized ``xywh`` boxes with shape ``[B, 4]``.
            template_length: number of template tokens.

        Returns:
            part_masks: float masks with shape ``[B, K, L_t]``.
            part_valid: whether each part owns at least one token, ``[B, K]``.
        """
        template_size = int(math.sqrt(template_length))
        if template_size * template_size != template_length:
            raise ValueError(
                "VDRM requires a square template token grid, "
                f"got {template_length} tokens"
            )

        bbox = template_bbox.to(dtype=torch.float32).clamp(0.0, 1.0)
        x0, y0, width, height = bbox.unbind(dim=-1)
        x1 = (x0 + width).clamp(0.0, 1.0)
        y1 = (y0 + height).clamp(0.0, 1.0)

        coord = (
            torch.arange(template_size, device=bbox.device, dtype=bbox.dtype)
            + 0.5
        ) / template_size
        grid_y, grid_x = torch.meshgrid(coord, coord, indexing="ij")
        grid_x = grid_x.flatten().view(1, 1, -1)
        grid_y = grid_y.flatten().view(1, 1, -1)

        part_masks = []
        for row in range(self.part_grid):
            part_y0 = y0 + (y1 - y0) * (row / self.part_grid)
            part_y1 = y0 + (y1 - y0) * ((row + 1) / self.part_grid)
            for col in range(self.part_grid):
                part_x0 = x0 + (x1 - x0) * (col / self.part_grid)
                part_x1 = x0 + (x1 - x0) * ((col + 1) / self.part_grid)
                mask = (
                    (grid_x >= part_x0[:, None, None])
                    & (grid_x < part_x1[:, None, None])
                    & (grid_y >= part_y0[:, None, None])
                    & (grid_y < part_y1[:, None, None])
                )
                part_masks.append(mask.squeeze(1))

        part_masks = torch.stack(part_masks, dim=1)
        part_valid = part_masks.any(dim=-1)
        return part_masks.to(dtype=template_bbox.dtype), part_valid

    def forward(
        self,
        tokens: torch.Tensor,
        template_length: int,
        template_bbox: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine search tokens and return reliability diagnostics.

        ``tokens`` must be ordered as template tokens followed by the currently
        retained search tokens, which is the default direct concatenation used
        by OSTrack-CE.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, L, C], got {tokens.shape}")
        if not 0 < template_length < tokens.shape[1]:
            raise ValueError(
                f"invalid template length {template_length} for {tokens.shape[1]} tokens"
            )

        batch_size = tokens.shape[0]
        if template_bbox is None:
            template_bbox = self._default_template_bbox(
                batch_size, tokens.device, tokens.dtype
            )
        else:
            template_bbox = template_bbox.reshape(batch_size, 4).to(
                device=tokens.device, dtype=tokens.dtype
            )

        template_tokens = tokens[:, :template_length]
        search_tokens = tokens[:, template_length:]
        part_masks, part_valid = self._build_part_masks(
            template_bbox, template_length
        )
        part_masks = part_masks.to(dtype=template_tokens.dtype)

        part_count = part_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        prototypes = torch.einsum(
            "bkl,blc->bkc", part_masks, template_tokens
        ) / part_count
        prototypes = prototypes * part_valid.unsqueeze(-1).to(prototypes.dtype)

        normalized_prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)
        normalized_search = F.normalize(search_tokens, dim=-1, eps=self.eps)
        similarity = torch.einsum(
            "bkc,blc->bkl", normalized_prototypes, normalized_search
        )

        match_topk = min(self.topk, search_tokens.shape[1])
        top_similarity = similarity.topk(match_topk, dim=-1).values.mean(dim=-1)
        match_scale = F.softplus(self.log_match_scale)
        part_reliability = torch.sigmoid(
            match_scale * top_similarity + self.match_bias
        )
        part_reliability = (
            part_reliability * part_valid.to(part_reliability.dtype)
        )

        part_attention = torch.softmax(similarity, dim=1)
        weighted_prototypes = (
            prototypes * part_reliability.unsqueeze(-1)
        )
        reconstructed = torch.einsum(
            "bkl,bkc->blc", part_attention, weighted_prototypes
        )
        token_match = similarity.max(dim=1).values.clamp_min(0.0)
        residual = reconstructed * token_match.unsqueeze(-1)

        template_tokens_out = template_tokens
        search_tokens_out = search_tokens + self.alpha * residual
        output_tokens = torch.cat(
            [template_tokens_out, search_tokens_out], dim=1
        )

        valid_count = part_valid.sum(dim=-1).clamp_min(1)
        visual_reliability = part_reliability.sum(dim=-1) / valid_count
        visual_reliability = torch.where(
            part_valid.any(dim=-1),
            visual_reliability,
            torch.zeros_like(visual_reliability),
        )

        diagnostics = {
            "part_reliability": part_reliability,
            "part_valid": part_valid,
            "visual_reliability": visual_reliability,
            "vdrm_alpha": self.alpha,
        }
        return output_tokens, diagnostics
