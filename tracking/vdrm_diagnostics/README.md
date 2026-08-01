# VDRM Backend 配对诊断

该目录只包含只读诊断工具。诊断不会修改模型权重、训练配置、标准
Tracker 状态转移或已有测试结果。

`diagnose_vdrm_backend_pairs.py` 对同一模板和同一个搜索裁剪分别运行重训
OSTrack 与 VDRM Backend，用于把以下因素拆开：

- 同输入下的视觉 Backend 差异；
- 独立闭环运行产生的搜索区域反馈；
- VDRM 可靠度能否区分正确和错误视觉结果；
- 中心变化和尺度变化分别造成的影响；
- VDRM 残差在搜索 token 上的空间分布。

正式服务器命令和输出字段见脚本的 `--help`。默认输出目录为：

```text
<save_dir>/vdrm_backend_pairs/<baseline>__vs__<vdrm>/<dataset>/
```

服务器先运行一个短 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tracking/vdrm_diagnostics/diagnose_vdrm_backend_pairs.py \
  --baseline-config vitb_256_mae_ce_32x4_ep300_fulltn \
  --vdrm-config vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300 \
  --dataset-name uav123 \
  --sequences uav_car7 \
  --anchor-mode baseline_replay \
  --max-frames 10 \
  --device cuda
```

短测试通过后去掉 `--max-frames`，并一次传入需要诊断的全部序列。关键
序列建议分别运行 `baseline_replay` 和 `ground_truth`；前者检测 Backend
替换兼容性，后者排除历史搜索区域漂移。
