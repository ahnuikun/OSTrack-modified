from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ..data.vdrm_augmentation import apply_structured_target_occlusion


def compute_vdrm_part_rank_loss(
    part_similarity,
    search_global_index,
    gaussian_map,
    part_valid=None,
    part_weight=None,
    margin=0.1,
):
    """Rank each template part's target match above background matches.

    ``part_similarity`` is computed inside VDRM before the residual is added.
    Candidate elimination may have removed search tokens, so
    ``search_global_index`` maps the retained tokens back to the original
    score-map grid used by ``gaussian_map``.
    """
    if part_similarity.ndim != 3:
        raise ValueError(
            "part_similarity must have shape [B, K, L], got "
            f"{tuple(part_similarity.shape)}"
        )
    batch_size, num_parts, search_length = part_similarity.shape
    if search_global_index.shape != (batch_size, search_length):
        raise ValueError(
            "search_global_index must have shape [B, L], got "
            f"{tuple(search_global_index.shape)}"
        )
    if gaussian_map.ndim != 3 or gaussian_map.shape[0] != batch_size:
        raise ValueError(
            "gaussian_map must have shape [B, H, W], got "
            f"{tuple(gaussian_map.shape)}"
        )

    global_index = search_global_index.to(
        device=part_similarity.device, dtype=torch.long
    )
    flat_gaussian = gaussian_map.to(
        device=part_similarity.device, dtype=part_similarity.dtype
    ).flatten(1)
    if (
        global_index.numel() == 0
        or global_index.min() < 0
        or global_index.max() >= flat_gaussian.shape[1]
    ):
        raise ValueError("search_global_index is outside gaussian_map")

    retained_gaussian = flat_gaussian.gather(1, global_index)
    positive_mask = retained_gaussian > 0.0

    # With aggressive candidate elimination the Gaussian support can be
    # removed. Use the retained token nearest to the target peak as a stable
    # fallback instead of dropping the whole sample.
    missing_positive = ~positive_mask.any(dim=1)
    if missing_positive.any():
        full_peak_index = flat_gaussian.argmax(dim=1)
        grid_width = gaussian_map.shape[2]
        token_x = global_index.remainder(grid_width)
        token_y = global_index.div(grid_width, rounding_mode="floor")
        peak_x = full_peak_index.remainder(grid_width).unsqueeze(1)
        peak_y = full_peak_index.div(
            grid_width, rounding_mode="floor"
        ).unsqueeze(1)
        squared_distance = (token_x - peak_x).square() + (
            token_y - peak_y
        ).square()
        fallback_index = squared_distance.argmin(dim=1, keepdim=True)
        fallback_mask = torch.zeros_like(positive_mask)
        fallback_mask.scatter_(1, fallback_index, True)
        positive_mask = positive_mask | (
            missing_positive.unsqueeze(1) & fallback_mask
        )

    negative_mask = (retained_gaussian <= 0.0) & ~positive_mask
    positive_similarity = part_similarity.masked_fill(
        ~positive_mask[:, None, :], -torch.inf
    ).amax(dim=-1)
    negative_similarity = part_similarity.masked_fill(
        ~negative_mask[:, None, :], -torch.inf
    ).amax(dim=-1)

    valid = negative_mask.any(dim=1, keepdim=True).expand(
        batch_size, num_parts
    )
    if part_valid is not None:
        valid = valid & part_valid.to(device=valid.device, dtype=torch.bool)

    element_loss = F.softplus(
        float(margin) + negative_similarity - positive_similarity
    )
    loss_weight = valid.to(element_loss.dtype)
    if part_weight is not None:
        if part_weight.shape != (batch_size, num_parts):
            raise ValueError(
                "part_weight must have shape [B, K], got "
                f"{tuple(part_weight.shape)}"
            )
        loss_weight = loss_weight * part_weight.to(
            device=element_loss.device, dtype=element_loss.dtype
        ).clamp(0.0, 1.0)

    return (element_loss * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        template_bbox = data['template_anno'][0].view(-1, 4)

        visibility_target = None
        visibility_applied = None
        vdrm_cfg = getattr(self.cfg.MODEL, "VDRM", None)
        vdrm_enabled = bool(vdrm_cfg is not None and vdrm_cfg.ENABLED)
        occlusion_probability = getattr(
            self.cfg.DATA.SEARCH, "VDRM_OCCLUSION_PROBABILITY", 0.0
        )
        if vdrm_enabled and self.net.training and occlusion_probability > 0.0:
            search_bbox = data['search_anno'][0].view(-1, 4)
            search_img, visibility_target, visibility_applied = (
                apply_structured_target_occlusion(
                    search_img,
                    search_bbox,
                    probability=occlusion_probability,
                    min_area_ratio=self.cfg.DATA.SEARCH.VDRM_OCCLUSION_MIN_RATIO,
                    max_area_ratio=self.cfg.DATA.SEARCH.VDRM_OCCLUSION_MAX_RATIO,
                    part_grid=int(vdrm_cfg.NUM_PARTS ** 0.5),
                )
            )

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            template_bbox=template_bbox,
                            return_last_attn=False)
        if visibility_target is not None:
            out_dict['vdrm_visibility_target'] = visibility_target
            out_dict['vdrm_visibility_applied'] = visibility_applied

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        visibility_loss = pred_boxes.new_zeros(())
        rank_loss = pred_boxes.new_zeros(())
        aux_weight_scale = 0.0

        vdrm_cfg = getattr(self.cfg.MODEL, "VDRM", None)
        vdrm_enabled = bool(vdrm_cfg is not None and vdrm_cfg.ENABLED)
        if vdrm_enabled:
            if (
                'part_reliability' in pred_dict
                and 'vdrm_visibility_target' in pred_dict
                and 'vdrm_visibility_applied' in pred_dict
            ):
                part_reliability = pred_dict['part_reliability']
                visibility_target = pred_dict['vdrm_visibility_target'].to(
                    device=part_reliability.device,
                    dtype=part_reliability.dtype,
                )
                visibility_mask = pred_dict['vdrm_visibility_applied'][:, None]
                if 'part_valid' in pred_dict:
                    visibility_mask = visibility_mask & pred_dict['part_valid']
                visibility_element = F.binary_cross_entropy(
                    part_reliability.clamp(1e-6, 1.0 - 1e-6),
                    visibility_target,
                    reduction='none',
                )
                visibility_mask_float = visibility_mask.to(
                    visibility_element.dtype
                )
                visibility_loss = (
                    visibility_element * visibility_mask_float
                ).sum() / visibility_mask_float.sum().clamp_min(1.0)

            reliability_mode = getattr(vdrm_cfg, 'RELIABILITY_MODE', 'topk')
            if reliability_mode == 'margin':
                required_rank_outputs = {
                    'part_similarity', 'search_global_index'
                }
                missing_rank_outputs = required_rank_outputs.difference(
                    pred_dict
                )
                if missing_rank_outputs:
                    raise RuntimeError(
                        "VDRM margin rank loss is missing model outputs: "
                        f"{sorted(missing_rank_outputs)}"
                    )
                part_weight = pred_dict.get('vdrm_visibility_target')
                rank_loss = compute_vdrm_part_rank_loss(
                    pred_dict['part_similarity'],
                    pred_dict['search_global_index'],
                    gt_gaussian_maps.squeeze(1),
                    part_valid=pred_dict.get('part_valid'),
                    part_weight=part_weight,
                    margin=getattr(
                        self.cfg.TRAIN, 'VDRM_RANK_MARGIN', 0.1
                    ),
                )
            elif 'score_logits' in pred_dict:
                score_logits = pred_dict['score_logits'].squeeze(1)
                gaussian_map = gt_gaussian_maps.squeeze(1)
                positive_index = gaussian_map.flatten(1).argmax(
                    dim=1, keepdim=True
                )
                positive_logit = score_logits.flatten(1).gather(
                    dim=1, index=positive_index
                ).squeeze(1)

                negative_mask = gaussian_map <= 0.0
                has_negative = negative_mask.flatten(1).any(dim=1)
                negative_logit = score_logits.masked_fill(
                    ~negative_mask, -torch.inf
                ).flatten(1).amax(dim=1)
                if has_negative.any():
                    rank_loss = F.softplus(
                        negative_logit[has_negative]
                        - positive_logit[has_negative]
                    ).mean()

            warmup_epochs = self.cfg.TRAIN.VDRM_AUX_WARMUP_EPOCHS
            epoch = float(gt_dict.get('epoch', 1))
            if warmup_epochs <= 1:
                aux_weight_scale = 1.0
            else:
                aux_weight_scale = min(
                    1.0, max(0.0, (epoch - 1.0) / (warmup_epochs - 1.0))
                )

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if vdrm_enabled:
            loss = loss + aux_weight_scale * (
                self.cfg.TRAIN.VDRM_VISIBILITY_WEIGHT * visibility_loss
                + self.cfg.TRAIN.VDRM_RANK_WEIGHT * rank_loss
            )
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if vdrm_enabled:
                status.update({
                    "Loss/vdrm_visibility": visibility_loss.item(),
                    "Loss/vdrm_rank": rank_loss.item(),
                    "VDRM/aux_weight_scale": aux_weight_scale,
                    "VDRM/alpha": pred_dict['vdrm_alpha'].detach().item()
                    if 'vdrm_alpha' in pred_dict else 0.0,
                    "VDRM/reliability": pred_dict['visual_reliability'].detach().mean().item()
                    if 'visual_reliability' in pred_dict else 0.0,
                })
                if "vdrm_distractor_applied" in gt_dict:
                    status["VDRM/distractor_applied_rate"] = (
                        gt_dict["vdrm_distractor_applied"]
                        .detach()
                        .float()
                        .mean()
                        .item()
                    )
                residual_status_keys = {
                    "vdrm_residual_clip_rate": "VDRM/residual_clip_rate",
                    "vdrm_raw_delta_relative_norm": (
                        "VDRM/raw_delta_relative_norm"
                    ),
                    "vdrm_delta_relative_norm": (
                        "VDRM/delta_relative_norm"
                    ),
                }
                for output_key, status_key in residual_status_keys.items():
                    if output_key in pred_dict:
                        status[status_key] = (
                            pred_dict[output_key].detach().mean().item()
                        )
                if 'part_match_margin' in pred_dict:
                    status.update({
                        "VDRM/match_margin": pred_dict[
                            'part_match_margin'
                        ].detach().mean().item(),
                        "VDRM/peak_similarity": pred_dict[
                            'part_peak_similarity'
                        ].detach().mean().item(),
                        "VDRM/hard_negative_similarity": pred_dict[
                            'part_hard_negative_similarity'
                        ].detach().mean().item(),
                    })
            return loss, status
        else:
            return loss
