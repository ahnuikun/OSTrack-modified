import unittest

import numpy as np
import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.train.actors.ostrack import (
    compute_vdrm_part_rank_loss,
    compute_vdrm_response_rank_loss,
)
from lib.train.data.sampler import TrackingSampler
from lib.train.data.vdrm_augmentation import (
    apply_same_class_distractor_copy_paste,
    apply_structured_target_occlusion,
)


class VDRMTest(unittest.TestCase):
    class _FakeClassDataset:
        def has_class_info(self):
            return True

        def get_sequences_in_class(self, class_name):
            return [0, 1] if class_name == "car" else []

        def is_video_sequence(self):
            return True

        def get_sequence_info(self, seq_id):
            bbox = torch.tensor([[8.0, 8.0, 16.0, 16.0]]).repeat(20, 1)
            visible = torch.ones(20, dtype=torch.uint8)
            return {"bbox": bbox, "visible": visible, "valid": visible}

        def get_frames(self, seq_id, frame_ids, anno=None):
            frames = [
                np.full((32, 32, 3), seq_id, dtype=np.uint8)
                for _ in frame_ids
            ]
            boxes = [anno["bbox"][frame_id].clone() for frame_id in frame_ids]
            return frames, {"bbox": boxes}, {"object_class_name": "car"}

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

    def test_relative_norm_bound_caps_the_complete_residual_update(self):
        torch.manual_seed(11)
        max_ratio = 0.25
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
            residual_max_ratio=max_ratio,
        )
        module.alpha.data.fill_(-20.0)
        tokens = torch.randn(2, 64 + 37, 32)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )

        search_input = tokens[:, 64:]
        search_delta = output[:, 64:] - search_input
        relative_norm = (
            torch.linalg.vector_norm(search_delta, dim=-1)
            / torch.linalg.vector_norm(search_input, dim=-1).clamp_min(1e-6)
        )
        self.assertLessEqual(relative_norm.max().item(), max_ratio + 1e-5)
        self.assertGreater(
            diagnostics["vdrm_residual_clip_rate"].item(), 0.0
        )
        self.assertGreater(
            diagnostics["vdrm_raw_delta_relative_norm"].item(),
            diagnostics["vdrm_delta_relative_norm"].item(),
        )

    def test_disabled_relative_norm_bound_preserves_v1_forward(self):
        torch.manual_seed(13)
        default_module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
        )
        explicit_v1_module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
            residual_max_ratio=0.0,
        )
        explicit_v1_module.load_state_dict(default_module.state_dict())
        default_module.alpha.data.fill_(-1.046)
        explicit_v1_module.alpha.data.copy_(default_module.alpha.data)
        tokens = torch.randn(2, 64 + 37, 32)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        default_output, _ = default_module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )
        explicit_output, _ = explicit_v1_module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )

        self.assertTrue(torch.equal(default_output, explicit_output))

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

    def test_response_rank_uses_pasted_distractor_for_applied_samples(self):
        score_logits = torch.zeros(2, 4, 4, requires_grad=True)
        with torch.no_grad():
            score_logits[:, 1, 1] = 2.0
            score_logits[0, 0, 0] = 1.0
            score_logits[:, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[:, 1, 1] = 1.0
        distractor_boxes = torch.tensor(
            [
                [0.00, 0.00, 0.25, 0.25],
                [0.00, 0.00, 0.00, 0.00],
            ]
        )
        distractor_applied = torch.tensor([1.0, 0.0])

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
            distractor_boxes=distractor_boxes,
            distractor_applied=distractor_applied,
        )
        expected = (
            torch.nn.functional.softplus(torch.tensor(1.0 - 2.0))
            + torch.nn.functional.softplus(torch.tensor(5.0 - 2.0))
        ) / 2.0

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["alignment_success_rate"].item(), 1.0)
        self.assertEqual(diagnostics["distractor_rank_margin"].item(), 1.0)
        loss.backward()
        self.assertNotEqual(score_logits.grad[0, 0, 0].item(), 0.0)
        self.assertEqual(score_logits.grad[0, 3, 3].item(), 0.0)

    def test_response_rank_without_alignment_preserves_global_negative(self):
        score_logits = torch.zeros(1, 4, 4)
        score_logits[0, 1, 1] = 2.0
        score_logits[0, 0, 0] = 1.0
        score_logits[0, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[0, 1, 1] = 1.0

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
        )

        expected = torch.nn.functional.softplus(torch.tensor(5.0 - 2.0))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["alignment_success_rate"].item(), 0.0)

    def test_response_rank_maps_tiny_distractor_to_nearest_cell(self):
        score_logits = torch.zeros(1, 4, 4)
        score_logits[0, 2, 2] = 2.0
        score_logits[0, 0, 0] = 1.0
        score_logits[0, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[0, 2, 2] = 1.0

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
            distractor_boxes=torch.tensor([[0.01, 0.01, 0.01, 0.01]]),
            distractor_applied=torch.tensor([1.0]),
        )

        expected = torch.nn.functional.softplus(torch.tensor(1.0 - 2.0))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["alignment_success_rate"].item(), 1.0)

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

    def test_same_class_sampler_uses_a_different_instance(self):
        sampler = TrackingSampler(
            datasets=[],
            p_datasets=[],
            samples_per_epoch=1,
            max_gap=10,
            num_search_frames=1,
            same_class_distractor_probability=1.0,
        )
        distractor = sampler._sample_same_class_distractor(
            self._FakeClassDataset(),
            source_seq_id=0,
            class_name="car",
        )

        self.assertIsNotNone(distractor)
        self.assertTrue((distractor["vdrm_distractor_images"][0] == 1).all())
        self.assertIsNone(
            sampler._sample_same_class_distractor(
                self._FakeClassDataset(),
                source_seq_id=0,
                class_name="Unknown",
            )
        )

    def test_same_class_copy_paste_preserves_target_pixels(self):
        torch.manual_seed(7)
        image = torch.zeros(3, 64, 64)
        distractor = torch.ones(3, 64, 64)
        target_box = torch.tensor([0.375, 0.375, 0.25, 0.25])
        distractor_box = torch.tensor([0.25, 0.25, 0.50, 0.50])

        augmented, applied, pasted_box = apply_same_class_distractor_copy_paste(
            image,
            target_box,
            distractor,
            distractor_box,
            min_scale=1.0,
            max_scale=1.0,
            invalid_mask=torch.zeros(64, 64, dtype=torch.bool),
        )

        self.assertTrue(applied)
        self.assertGreater(augmented.count_nonzero().item(), 0)
        self.assertTrue(torch.equal(augmented[:, 24:40, 24:40], image[:, 24:40, 24:40]))
        self.assertTrue((pasted_box >= 0.0).all())
        self.assertTrue((pasted_box <= 1.0).all())
        self.assertGreater(pasted_box[2].item(), 0.0)
        self.assertGreater(pasted_box[3].item(), 0.0)

        paste_x0 = int(round(pasted_box[0].item() * 64))
        paste_y0 = int(round(pasted_box[1].item() * 64))
        paste_x1 = int(round((pasted_box[0] + pasted_box[2]).item() * 64))
        paste_y1 = int(round((pasted_box[1] + pasted_box[3]).item() * 64))
        overlaps_target = not (
            paste_x1 <= 24
            or paste_x0 >= 40
            or paste_y1 <= 24
            or paste_y0 >= 40
        )
        self.assertFalse(overlaps_target)


if __name__ == "__main__":
    unittest.main()
