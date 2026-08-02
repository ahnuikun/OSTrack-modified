# VDRM-v7 Candidate-Consistent Spatial Gate

## Registered hypothesis

The epoch-300 V3 diagnosis shows positive mean backend deltas under all four
shared-crop conditions, while standard closed-loop evaluation contains very
large positive and negative per-sequence AUC bifurcations. Catastrophic paired
frames are dominated by center error, and residual energy is not larger on
failed frames. The registered V7 hypothesis is therefore:

> A residual enabled by isolated or spatially scattered part matches can seed
> a wrong center candidate; tracker feedback amplifies that small error into a
> different trajectory.

V7 tests this hypothesis without changing HNCP distance, probability, source
sampling, response-rank supervision, training datasets, or M2.

## Controlled change

V3 uses the strongest positive part match independently at every search token.
V7 adds `SPATIAL_GATE_MODE: candidate_consensus`:

1. Scatter retained CE-token similarities to the original search grid.
2. For each template part, take the local maximum in a fixed 3x3 neighborhood.
3. Require the strongest two parts to agree around the same candidate.
4. Calibrate that evidence into a candidate reliability map.
5. Multiply the V3 residual by this spatial map before applying `alpha`.

The existing `visual_reliability` remains unchanged. V7 additionally exposes
`candidate_target_reliability`, sampled at the final Center Head peak.

The candidate map receives a CenterNet-style focal objective on retained CE
tokens. If CE removes the exact target-center token, the nearest retained token
is used as the positive fallback. The loss follows the existing VDRM auxiliary
warm-up schedule.

## Compatibility

- `SPATIAL_GATE_MODE: token_match` is the default and preserves V1-V6 behavior.
- Candidate calibration parameters exist only in `candidate_consensus` mode,
  so old V3 checkpoint state dictionaries do not gain missing keys.
- `alpha` remains zero initialized.
- `RESIDUAL_MAX_RATIO` remains disabled for V7.
- Existing HNCP parameters and the V3 response-rank loss remain unchanged.

## Experiment configuration

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_v7_ccsg_hncp_32x4_ep300.yaml
```

Training-only controlled settings:

```yaml
MODEL:
  VDRM:
    SPATIAL_GATE_MODE: candidate_consensus
    CANDIDATE_LOCAL_RADIUS: 1
    CANDIDATE_CONSENSUS_PARTS: 2
TRAIN:
  VDRM_CANDIDATE_WEIGHT: 0.5
```

These values are registered before V7 training and must not be tuned on
UAV123 or DTB70 test results.

## Required training observations

- `Loss/vdrm_candidate`
- `VDRM/candidate_target_reliability`
- `VDRM/candidate_reliability_mean`
- `VDRM/alpha`
- `VDRM/reliability`
- `VDRM/distractor_applied_rate`
- `Loss/vdrm_rank`

Non-finite loss, a permanently zero candidate gradient, or a candidate map
that saturates globally near zero/one invalidates the run.

## Decision Gate

The epoch-300 model is accepted only if all conditions hold:

1. UAV123 AUC is at least baseline +0.30.
2. DTB70 AUC is at least baseline +0.30.
3. Full paired mean delta IoU is non-negative for UAV123/DTB70 under both
   `ground_truth` and `baseline_replay`.
4. Catastrophic paired frames and center-only catastrophic losses decrease
   relative to V3.
5. Candidate reliability is reported separately from visual reliability; M2
   integration cannot replace the independent module-one Gate.

If paired tails improve but closed-loop AUC still bifurcates, the next isolated
experiment may add dual-crop consistency training. It must not be mixed into
the first V7 run.
