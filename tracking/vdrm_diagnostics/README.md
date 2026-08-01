# VDRM Backend 配对诊断

本目录提供只读诊断工具。配对诊断不会修改模型权重、训练配置、标准
Tracker 状态转移或已有测试结果。

`diagnose_vdrm_backend_pairs.py` 将同一模板和同一搜索裁剪分别送入
OSTrack 与 VDRM Backend，用于区分视觉 Backend 差异和闭环搜索反馈。

## 一键执行正式计划

以下入口会依次完成：

1. 单元测试和 10 帧 smoke；
2. UAV123 四个登记关键序列的两种 anchor 模式；
3. 使用显式 epoch-300 checkpoint 生成 UAV123、DTB70 正式闭环结果；
4. 按逐序列 AUC 自动选择下降 10、近似不变 5、提升 5；
5. 对两个数据集运行 `ground_truth` 和 `baseline_replay`；
6. 验证所有 CSV/JSON 并生成五项证据和模块一决策报告。

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  tracking/vdrm_diagnostics/run_vdrm_diagnostic_plan.py \
  --baseline-checkpoint \
  /home/professor12/OSTrack-main/output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300_fulltn/OSTrack_ep0300.pth.tar \
  --vdrm-checkpoint \
  /home/professor12/OSTrack-main/output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300/OSTrack_ep0300.pth.tar \
  --run-formal-auc \
  --gpu-id 0
```

结果写入时间戳目录：

```text
output/vdrm_backend_pairs/ep0300_<timestamp>/
```

主要交付文件：

- `run_manifest.json`：配置、显式 checkpoint、SHA-256 和正式 AUC run ID；
- `selection/*_selection.csv`：10/5/5 选择依据；
- `<dataset>/<anchor_mode>/dataset_summary.json`：配对诊断汇总；
- `diagnostic_metrics.csv`：视觉、中心、尺度、可靠度、残差五项证据；
- `largest_loss_manifest.json`：各数据集/模式最大下降序列及 CSV；
- `diagnostic_report.md`：根因信号和模块一优化方向。

已有正式结果可以复用：

```bash
python -u tracking/vdrm_diagnostics/run_vdrm_diagnostic_plan.py \
  --baseline-checkpoint /path/to/baseline.pth.tar \
  --vdrm-checkpoint /path/to/v3.pth.tar \
  --use-existing-auc-results \
  --auc-run-id 123456 \
  --gpu-id 0
```

如果已有结果位于无 run ID 的默认目录，省略 `--auc-run-id`。只有确认这些
结果确实来自目标 checkpoint 时才应复用。

加上 `--full-paired` 会额外对 UAV123 和 DTB70 全部序列运行两种配对模式。
配对结果中的 `mean_delta_iou` 不是正式闭环 AUC，两类结果不能互相替代。
