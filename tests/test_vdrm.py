import unittest

import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.train.actors.ostrack import compute_vdrm_part_rank_loss
from lib.train.data.vdrm_augmentation import (
    apply_structured_target_occlusion,
)


class VDRMTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_tokens(self):
        torch.manual_seed(0)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
        )
        tokens = torch.randn(2, 64 + 37, 32)
        template_bbox = torch.tensor(
            [
                [0.25, 0.25, 0.50, 0.50],
                [0.20, 0.20, 0.60, 0.60],
            ]
        )

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["part_reliability"].shape, (2, 4))
        self.assertEqual(diagnostics["part_valid"].shape, (2, 4))
        self.assertEqual(diagnostics["visual_reliability"].shape, (2,))
        self.assertTrue(diagnostics["part_valid"].all())
        self.assertTrue(
            ((diagnostics["visual_reliability"] >= 0.0)
             & (diagnostics["visual_reliability"] <= 1.0)).all()
        )

    def test_gradients_reach_residual_and_reliability_parameters(self):
        torch.manual_seed(1)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
        )
        tokens = torch.randn(2, 64 + 37, 32, requires_grad=True)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )
        loss = output.square().mean() + diagnostics[
            "visual_reliability"
        ].mean()
        loss.backward()

        self.assertIsNotNone(module.alpha.grad)
        self.assertIsNotNone(module.log_match_scale.grad)
        self.assertIsNotNone(module.match_bias.grad)
        self.assertTrue(torch.isfinite(module.alpha.grad))
        self.assertTrue(torch.isfinite(module.log_match_scale.grad))
        self.assertTrue(torch.isfinite(module.match_bias.grad))
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_margin_reliability_suppresses_the_first_peak_neighborhood(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            reliability_mode="margin",
            nms_radius=1,
        )
        similarity = torch.tensor(
            [[[0.90, 0.85, 0.70, 0.80, 0.60]]]
        )
        global_index = torch.tensor([[0, 1, 2, 5, 15]])

        peak, hard_negative, margin = module._margin_statistics(
            similarity,
            global_index,
            search_grid_size=4,
        )

        self.assertTrue(torch.allclose(peak, torch.tensor([[0.90]])))
        # Locations 1 and 5 belong to the 3x3 neighborhood around location 0.
        self.assertTrue(
            torch.allclose(hard_negative, torch.tensor([[0.70]]))
        )
        self.assertTrue(torch.allclose(margin, torch.tensor([[0.20]])))

    def test_margin_forward_preserves_zero_initialized_residual(self):
        torch.manual_seed(3)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            reliability_mode="margin",
            nms_radius=1,
            initial_match_bias=0.0,
        )
        tokens = torch.randn(2, 64 + 37, 32)
        global_index = torch.arange(37).unsqueeze(0).repeat(2, 1)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=global_index,
            search_grid_size=16,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["part_similarity"].shape, (2, 4, 37))
        self.assertEqual(diagnostics["part_match_margin"].shape, (2, 4))
        self.assertTrue((diagnostics["part_match_margin"] >= 0.0).all())

    def test_part_rank_loss_rewards_target_over_background(self):
        good_similarity = torch.tensor(
            [[[0.90, 0.80, 0.10, 0.20],
              [0.70, 0.60, 0.30, 0.10]]],
            requires_grad=True,
        )
        bad_similarity = torch.tensor(
            [[[0.10, 0.20, 0.90, 0.80],
              [0.20, 0.10, 0.70, 0.60]]]
        )
        global_index = torch.tensor([[0, 1, 4, 15]])
        gaussian_map = torch.zeros(1, 4, 4)
        gaussian_map[0, 0, :2] = 1.0
        part_valid = torch.ones(1, 2, dtype=torch.bool)

        good_loss = compute_vdrm_part_rank_loss(
            good_similarity,
            global_index,
            gaussian_map,
            part_valid=part_valid,
            margin=0.1,
        )
        bad_loss = compute_vdrm_part_rank_loss(
            bad_similarity,
            global_index,
            gaussian_map,
            part_valid=part_valid,
            margin=0.1,
        )

        self.assertLess(good_loss.item(), bad_loss.item())
        good_loss.backward()
        self.assertIsNotNone(good_similarity.grad)
        self.assertTrue(torch.isfinite(good_similarity.grad).all())

    def test_structured_occlusion_returns_soft_part_labels(self):
        torch.manual_seed(2)
        images = torch.ones(2, 3, 32, 32)
        boxes = torch.tensor(
            [
                [0.25, 0.25, 0.50, 0.50],
                [0.20, 0.20, 0.60, 0.60],
            ]
        )

        occluded, visibility, applied = apply_structured_target_occlusion(
            images,
            boxes,
            probability=1.0,
            min_area_ratio=0.3,
            max_area_ratio=0.3,
            part_grid=2,
        )

        self.assertTrue(applied.all())
        self.assertEqual(visibility.shape, (2, 4))
        self.assertTrue((visibility >= 0.0).all())
        self.assertTrue((visibility <= 1.0).all())
        self.assertTrue((visibility < 1.0).any(dim=1).all())
        self.assertGreater((occluded == 0.0).sum().item(), 0)

        unchanged, clean_visibility, clean_applied = (
            apply_structured_target_occlusion(
                images,
                boxes,
                probability=0.0,
                part_grid=2,
            )
        )
        self.assertTrue(torch.equal(unchanged, images))
        self.assertTrue((clean_visibility == 1.0).all())
        self.assertFalse(clean_applied.any())


if __name__ == "__main__":
    unittest.main()
