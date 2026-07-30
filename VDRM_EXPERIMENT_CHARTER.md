# VDRM-OSTrack 实验计划与目标

> 状态：项目主线约束（后续实现、训练、消融和汇报均以本文档为准）
> 当前项目范围：仅实现模块一 VDRM；模块二 CTMP 在另一项目中独立实现
> 开发环境：本机单张 RTX 5070，只做无数据集依赖的最小可运行验证
> 完整训练环境：服务器 4× RTX 4090；服务器数据集路径保持现状，不在本机修改
> 基线：OSTrack ViT-Base + CE，256×256 search，MAE ViT-Base 初始化

## 1. 总目标

本项目只研究并实现 **VDRM（Visibility-guided Discriminative Representation Module，遮挡—干扰鲁棒表征模块）**，目标是在不引入运动模型、不改变训练数据集范围的前提下，增强 OSTrack 对以下视觉困难的鲁棒性：

- 局部或完全遮挡；
- 背景干扰和相似目标；
- 低分辨率目标；
- 遮挡与干扰共同出现时的错误响应。

VDRM 应做到：

1. 将模板目标表示为若干部件原型；
2. 根据模板部件与搜索 token 的匹配估计部件可靠性；
3. 降低遮挡部件权重，提高可见部件权重；
4. 从搜索区域中选择最高错误响应作为硬干扰负样本；
5. 约束真实目标响应高于相似干扰目标响应；
6. 保持与原始 OSTrack Tracker Backend 的输入、输出和测试流程兼容。

项目最终需要证明：VDRM 本身能够提升视觉表征鲁棒性，并且可以在不重写 CTMP 的情况下替换官方 OSTrack Backend。

## 2. 两模块边界

| 项目 | 模块一：VDRM（本项目） | 模块二：CTMP（外部项目） |
|---|---|---|
| 核心问题 | 遮挡、干扰和相似目标下的视觉表征 | 相机运动补偿与目标残余运动预测 |
| 所在阶段 | 训练侧，推理时随 Tracker Backend 使用 | 纯推理侧、即插即用 |
| 是否训练 | 是 | 否 |
| 初始化 | MAE ViT-Base | 不适用 |
| 主要数据 | LaSOT、GOT-10k、COCO、TrackingNet | 使用已训练 Tracker 在测试视频上运行 |
| 本仓库是否实现 | 是 | 否 |
| 允许的集成方式 | 输出标准 `TrackerOutput` | 只消费标准输出，不反向侵入 VDRM |

### 2.1 本项目明确不做

以下内容不属于 VDRM，不得混入本项目主干或作为 VDRM 训练收益的一部分：

- ORB/SIFT 特征提取与背景 Mask；
- RANSAC 仿射矩阵或单应矩阵估计；
- 相机运动置信度；
- Kalman Filter、常速度或常加速度预测；
- 搜索中心运动重定位；
- 视觉—运动框融合；
- CTMP 阈值调优；
- 将 VisDrone、UAV123、UAVDT 或 DTB70 加入训练集。

如联调需要 CTMP，只允许通过统一接口接入；不得让 VDRM 的训练代码依赖 CTMP。

## 3. 已锁定的训练协议

除非实验结果证明某一项无法运行，并经过“变更控制”记录，否则以下设置不得随意修改。

### 3.1 训练数据

```yaml
DATASETS_NAME:
  - LASOT
  - GOT10K_vottrain
  - COCO17
  - TRACKINGNET

DATASETS_RATIO:
  - 1
  - 1
  - 1
  - 1

SAMPLE_PER_EPOCH: 60000
```

约束：

- 严格保持 OSTrack 原始四数据集；
- 不使用 VisDrone 参与训练；
- 结构化遮挡与同类目标 Copy-Paste 属于训练增强，不计作第二个网络模块；
- 增强策略必须可单独开关，以便消融并避免与 VDRM 结构收益混淆。

### 3.2 模型与优化

| 设置 | 锁定值 |
|---|---|
| Backbone | MAE ViT-Base 初始化 |
| Search size | 256×256 |
| 总 epoch | 300 |
| 学习率下降 epoch | 240 |
| Head 与 VDRM 学习率 | `4e-4` |
| Backbone 学习率 | `4e-5` |
| CE 位置（论文的一基编号） | Block 4、7、10 |
| CE 位置（当前配置的零基编号） | `CE_LOC: [3, 6, 9]` |
| VDRM 插入位置 | Block 6 输出后、进入 Block 7 前 |
| 本机开发硬件 | 单张 RTX 5070，仅做前向、反向和极短 smoke test |
| 服务器训练硬件 | 4× RTX 4090，执行完整 300-epoch 训练 |
| Checkpoint 与验证周期 | 每 20 epoch |

基准配置为：

```text
experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml
```

实现 VDRM 时应新建独立配置，不直接改写官方基准配置。

环境边界：

- 当前本机 `data/` 为空，不在本机运行真实数据集训练或评测；
- 项目内训练和测试路径是服务器上的正确路径，不为适配本机而修改；
- 本机测试只使用随机合成张量、人工框和已有本地权重（如需要）；
- 四卡 DDP、完整数据加载、300-epoch 训练及正式验证只能在服务器执行；
- 本机测试通过仅代表代码路径可运行，不代表服务器训练或最终精度已经验证。

### 3.3 损失函数

总损失固定为：

\[
\mathcal L =
\mathcal L_{\mathrm{OSTrack}}
+ \lambda_v \mathcal L_{\mathrm{visibility}}
+ \lambda_r \mathcal L_{\mathrm{rank}}
\]

其中：

\[
\mathcal L_{\mathrm{OSTrack}}
=
\mathcal L_{\mathrm{focal}}
+2\mathcal L_{\mathrm{GIoU}}
+5\mathcal L_1
\]

第一版固定使用：

\[
\lambda_v=0.5,\qquad \lambda_r=0.5
\]

前 20 epoch 对两个辅助损失做线性 warm-up：

\[
\lambda(e)=
\lambda_{\mathrm{target}}\cdot
\operatorname{clip}
\left(\frac{e-1}{19},0,1\right)
\]

其中训练 epoch 按 `1..300` 编号：第 1 epoch 权重为 0，第 20 epoch 达到目标值。实现时必须保持该定义，避免 off-by-one 问题。

### 3.4 零初始化残差

VDRM 采用：

\[
F'=F+\alpha F_{\mathrm{VDRM}},
\qquad \alpha_{\mathrm{init}}=0
\]

该设计用于保护 MAE 初始化。必须记录并验证：

- 初始化时带 VDRM 模型与原 OSTrack 前向结果一致或仅存在数值误差；
- `alpha` 可训练且能够离开 0；
- VDRM 分支在训练开始后能够获得非零梯度；
- 不因混合精度、分布式训练或参数分组遗漏而冻结。

## 4. VDRM 功能定义

### 4.1 部件原型与可靠性

- 从模板目标区域构建若干部件原型；
- 以部件原型和搜索 token 的匹配关系估计部件可靠性；
- 可靠性用于重加权部件或搜索特征；
- 遮挡部件应产生较低权重，可见且稳定的部件应产生较高权重。

部件数量、原型聚合方式、可靠性归一化方式属于可消融的实现细节，不得改变“可见性引导部件重加权”这一主假设。

### 4.2 硬干扰负样本

- 在搜索响应中排除真实目标邻域；
- 从剩余区域选择最高错误响应；
- 将该位置视为当前样本的硬干扰负样本；
- 排序损失约束真实目标响应高于硬干扰响应。

负样本排除范围、NMS 半径和 margin 属于待注册超参数。必须在完整训练前固定，不能依据最终测试集结果反复调节。

### 4.3 可见性监督

`L_visibility` 的监督来源和标签构造需要在实现前形成明确设计记录。无论采用结构化遮挡产生的已知 Mask、几何映射得到的部件可见性，还是其他训练期监督，都必须满足：

- 不需要测试集标注；
- 不引入第五个训练数据集；
- 训练与推理定义一致；
- 标签构造可复现；
- 能单独关闭以完成损失消融。

### 4.4 第一版 VDRM-v1 注册设计

第一版以“最少组件、最少参数、先跑通并观察有效信号”为原则，锁定以下实现，不加入额外质量头、多级门控或教师网络。

1. 在 Block 6 输出后、Block 7 输入前执行 VDRM。
2. 利用模板目标 Mask，将模板目标 token 按目标框局部坐标固定划分为 `2×2` 四个部件。
3. 每个有效部件通过 Masked Mean 得到一个部件原型，共最多四个原型。
4. 对部件原型和当前保留的搜索 token 计算余弦相似度。
5. 每个部件取搜索侧 Top-4 相似度均值，并通过一个共享标量仿射加 Sigmoid 得到部件可靠性。
6. 搜索 token 按部件相似度和部件可靠性重构模板部件残差。
7. 使用单个可学习残差系数 `alpha` 回注搜索 token，且 `alpha_init=0`。
8. `visual_reliability` 定义为有效部件可靠性的均值，语义是“当前帧可用的模板部件证据比例”，不是预测框 IoU 或完整成功概率。
9. 结构化遮挡仅作用于训练搜索目标区域，记录四个部件的未遮挡面积比例作为软标签。
10. `L_visibility` 只在具有已知合成遮挡标签的样本上计算。
11. `L_rank` 直接使用 Center Head 的 Sigmoid 前响应 logits：GT 中心为正响应，GT Gaussian 支持区域之外的最高响应为硬负响应。
12. 第一版排序损失使用无 margin 形式：

\[
\mathcal L_{\mathrm{rank}}
=
\operatorname{softplus}
\left(l_{\mathrm{neg}}-l_{\mathrm{pos}}\right)
\]

第一版不实现：

- 可学习部件 Query 或聚类；
- 多尺度、多层 VDRM；
- 通道门控与空间门控堆叠；
- IoU 质量预测 MLP；
- 动态 CE keep ratio；
- 教师—学生或双路特征一致性；
- 同类目标 Copy-Paste。

同类目标 Copy-Paste 仅保留为核心 VDRM 跑通后的独立训练增强消融，不属于第一版实现门禁。

### 4.5 第二版 VDRM-v2 注册设计

VDRM-v1 完整训练与四序列逐帧诊断表明，第一版绝对 Top-4 相似度可靠性不能稳定区分“目标身份匹配正确”和“搜索区中存在模板相似内容”。第二版只修正可靠性证据及其排序监督，不改部件原型、残差路径、插入位置、主损失或训练协议。

1. 继续使用模板目标局部 `2×2` 四部件、Masked Mean 原型和余弦相似度图。
2. 对每个部件找到搜索相似度第一峰；在原始 `16×16` 搜索 token 坐标系中抑制第一峰周围半径 1 的方形邻域，再选择空间上不同的最强峰作为硬干扰。
3. 部件可靠性证据改为非负匹配间隔：

\[
m_k=\max\left(0,s^+_k-s^-_k\right)
\]

4. 可靠性仍只使用共享标量仿射和 Sigmoid：

\[
q_k=\sigma\left(\operatorname{softplus}(a)m_k+b\right)
\]

不增加 MLP、质量预测头或额外门控。V2 初始化使用 `scale=5.0, bias=0.0`；偏置可通过可见性监督学习到低于 0.5 的遮挡可靠性。
5. `L_rank` 直接监督 VDRM 内部的部件—搜索 token 相似度图，而不是最终 Center Head 响应。GT Gaussian 支持区域内最高相似度为正样本，区域外最高相似度为硬负样本：

\[
\mathcal L_{\mathrm{rank}}
=
\operatorname{softplus}
\left(0.1+s^-_k-s^+_k\right)
\]

6. 若 CE 已删除 GT Gaussian 支持区的全部 token，则以当前保留 token 中距离 GT 中心最近者作为正样本，避免样本被静默丢弃。
7. 结构化遮挡样本的排序损失按部件可见比例加权，避免强迫完全遮挡部件匹配目标；普通样本权重为 1。
8. 继续使用单个零初始化 `alpha` 和原残差重构公式，不修改残差方向、强度或注入位置。
9. V1 的 `topk` 分支完整保留，旧配置与旧 checkpoint 可继续复现；V2 使用独立配置和独立输出目录。

第二版配置：

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_v2_32x4_ep300.yaml
```

VDRM-v2 已在完整训练后判定失败，只保留用于复现实验，不作为后续版本起点。最终 checkpoint 中匹配尺度收缩至约 `0.000169`、偏置约为 `0.619`，使可靠性几乎固定在 `0.65`；同时 `alpha` 下降至约 `-2.443`。五个统一测试集均未形成相对重训 OSTrack 的稳定收益。

### 4.6 第三版 VDRM-v3-HNCP 注册设计

VDRM-v3 回退到 VDRM-v1，只加入一项训练策略改动：真实同类硬干扰 Copy-Paste（Hard-Negative Copy-Paste, HNCP）。

1. 网络结构、Top-4 可靠性、残差公式、`alpha`、插入位置和 V1 的 Center Head `L_rank` 全部保持不变。
2. 从当前训练样本所属数据集的类别索引中，采样不同序列/实例的同语义类别可见目标；不跨数据集伪造类别映射。
3. 使用真实标注框裁出该实例，在搜索图中目标框外粘贴；粘贴区域不得覆盖真实目标或 padding，真实目标框和全部监督标签保持不变。
4. 训练样本采样概率固定为 `0.3`，干扰实例面积尺度相对真实目标使用 `[0.7, 1.3]`。类别未知或不存在另一实例时直接跳过，不用随机异类目标回退。
5. 已有 V1 `L_rank` 继续从最终 Center Head 响应图中选择最高错误响应，因此会自然监督新增干扰；不新增 loss、margin、门控、质量头或可学习参数。
6. 增强只在训练 dataloader 启用，验证、测试和 Tracker 接口完全不变。
7. 日志只增加 `VDRM/distractor_applied_rate`，用于确认真实应用率；该统计不参与前向或损失。

第三版配置：

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300.yaml
```

## 5. 统一 Tracker 接口

VDRM Tracker 必须提供以下逻辑输出：

```python
TrackerOutput = {
    "bbox": visual_bbox,
    "score": max_score,
    "response_map": score_map,
    "visual_reliability": reliability,  # 可选
}
```

接口约束：

- `bbox`、`score` 和 `response_map` 为必需字段；
- `visual_reliability` 为可选字段；
- 缺少 `visual_reliability` 时，下游 CTMP 必须仍可从 OSTrack 原始响应图计算视觉置信度；
- 替换官方 OSTrack 权重/Backend 后，字段语义、坐标系和响应图含义不得改变；
- VDRM 内部张量和辅助损失不得泄漏为 CTMP 的强依赖。

跨项目联调的 CTMP 输入约定为：

```python
MotionPluginInput = {
    "prev_frame": prev_frame,
    "current_frame": current_frame,
    "previous_bbox": previous_bbox,
    "tracker_output": tracker_output,
    "history": bbox_history,
}
```

本文只冻结该接口，不在本项目实现 `MotionPluginInput` 的处理逻辑。

## 6. 开发里程碑与门禁

### M0：基线冻结

- 保存原始 OSTrack 配置副本或通过继承生成 VDRM 配置；
- 保持现有服务器数据集和验证集路径不变；
- 本机不因 `data/` 为空而创建伪路径或改写服务器路径；
- 记录代码版本、环境、CUDA/PyTorch、随机种子与硬件；
- 在本机用随机合成张量跑通未修改 OSTrack 的前向和输出；
- 官方权重统一基线结果留到服务器或具备真实测试集的环境执行。

**本机通过条件：** 基线模型可构建、随机输入可前向，且没有修改任何服务器路径。
**服务器通过条件：** 基线可复现，且后续 VDRM 实验使用完全相同的测试协议。

### M1：VDRM 最小实现

- 在 Block 6 后插入 VDRM；
- 完成部件原型、可靠性估计和残差回注；
- 实现可选 `visual_reliability` 输出；
- 新建 VDRM 配置，不修改官方基准配置。

**通过条件：**

- `alpha=0` 时与原网络前向等价；
- 张量形状、设备和 dtype 正确；
- 单卡前向和反向无异常；
- 原始 OSTrack Loss 未被意外改变。

### M2：辅助损失与增强

- 实现 `L_visibility`；
- 实现硬负样本挖掘和 `L_rank`；
- 实现前 20 epoch 权重 warm-up；
- 将结构化遮挡做成独立配置开关；
- 同类目标 Copy-Paste 延后到核心模块证明可运行之后，不纳入第一版。

**通过条件：**

- 所有 Loss 为有限值；
- 硬负样本不落入真实目标排除区；
- 新模块、`alpha` 和对应损失路径均存在有效梯度；
- 关闭所有新增项后可退化为原始 OSTrack 行为。

### M3：训练 Smoke Test

本机仅按以下顺序执行：

1. 少量人工/合成 batch 的单元级检查；
2. CPU 前向与损失检查；
3. 单张 RTX 5070、batch size 1 的前向和反向检查；
4. 显存允许时执行极少量优化步，不运行完整 epoch。

**本机通过条件：**

- 无 NaN/Inf、OOM、DDP unused parameter 或 checkpoint 恢复错误；
- Loss、学习率、辅助权重、`alpha` 和梯度统计均被记录；
- VDRM 关闭或 `alpha=0` 时保持基线退化行为；
- 单卡前向、反向以及保存/恢复最小 checkpoint 可运行。

本机不得执行四卡测试。四卡 DDP、真实 dataloader 和 1-epoch smoke test属于服务器上传后的独立门禁。

### M4：完整训练

- 本阶段只在服务器执行；
- 4× RTX 4090；
- 完整训练 300 epoch；
- epoch 240 降低学习率；
- 每 20 epoch 保存 checkpoint；
- 每 20 epoch 保存验证结果；
- 训练中不依据测试集表现改变模型或超参数。

**通过条件：** 300 epoch 完成，最终权重、配置、日志、版本信息和验证曲线齐全。

### M5：统一测试与消融

统一测试：

- UAV123；
- UAVDT；
- DTB70；
- VisDrone-SOT test；
- LaSOT test。

所有模型必须使用相同的数据版本、初始化方式、测试代码、搜索设置和指标实现。不得为单个测试集定制 VDRM 参数。

## 7. 实验矩阵

### 7.1 论文主实验

| 实验 | 训练侧 VDRM | 推理侧 CTMP | 责任项目 |
|---|:---:|:---:|---|
| OSTrack |  |  | 本项目冻结基线 |
| OSTrack + VDRM | ✓ |  | 本项目主实验 |
| OSTrack + CTMP |  | ✓ | CTMP 项目 |
| OSTrack + VDRM + CTMP | ✓ | ✓ | 最终跨项目集成 |

本项目交付重点是前两行及 VDRM 权重；后两行不得阻塞 VDRM 开发。

### 7.2 VDRM 最小消融

至少保留以下可比较实验：

| 编号 | 原型/可靠性分支 | `L_visibility` | `L_rank` | 结构化遮挡 | 同类 Copy-Paste |
|---|:---:|:---:|:---:|:---:|:---:|
| A0 |  |  |  |  |  |
| A1 | ✓ |  |  |  |  |
| A2 | ✓ | ✓ |  |  |  |
| A3 | ✓ |  | ✓ |  |  |
| A4 | ✓ | ✓ | ✓ |  |  |
| A5 | ✓ | ✓ | ✓ | ✓ |  |
| A6 | ✓ | ✓ | ✓ | ✓ | ✓ |

其中 A0 是同协议 OSTrack 基线，A4 用于证明网络模块和两个辅助损失的核心贡献，A5/A6 用于区分训练增强收益。

如算力不足，可先用短训练筛除明显失效方案，但论文中的主比较必须采用相同的完整训练预算，不能将短训练结果与 300-epoch 结果直接比较。

## 8. 评价目标与归因原则

### 8.1 主要评价目标

- VDRM 在遮挡、背景干扰、相似目标和低分辨率属性上优于同协议 OSTrack；
- 整体性能提升不能只来自单一数据集或单一序列；
- 增益应由 A1–A4 消融证明来自 VDRM 结构及其损失，而非仅来自数据增强；
- 模型在无 CTMP 时必须独立工作；
- 与 CTMP 组合时无需改变 VDRM 权重或重写 CTMP。

### 8.2 测试集角色

| 测试集/属性 | 主要观察内容 |
|---|---|
| UAV123 | 快速运动、遮挡、尺度变化和无人机视角 |
| UAVDT | 相机运动、低分辨率、密集干扰 |
| DTB70 | 快速运动、相机运动、形变与遮挡 |
| VisDrone-SOT test | 小目标、密集同类目标、遮挡与背景干扰 |
| LaSOT test | 长时跟踪、完全遮挡、出视野和相似目标 |

VDRM 的论文归因重点是遮挡、背景干扰、相似目标和低分辨率。相机运动、搜索区域重定位和长时运动外推的收益应归因于 CTMP，不得将 CTMP 收益写成 VDRM 收益。

### 8.3 指标与报告

每次正式测试至少记录：

- Success/AUC；
- Precision；
- Normalized Precision（数据集支持时）；
- 相对同协议 OSTrack 的绝对差值；
- 速度、参数量和 FLOPs；
- 失败序列与对应属性；
- 所用 checkpoint、配置、代码版本和测试命令。

正式测试的数值成功阈值应在完整训练前登记到实验记录中，并对所有候选模型保持一致；不得看到最终测试结果后反向修改成功阈值。

## 9. 风险与控制

| 风险 | 控制措施 |
|---|---|
| VDRM 破坏 MAE 初始化 | 零初始化残差；先做前向等价测试 |
| `alpha=0` 导致分支早期无有效学习 | 记录 `alpha` 与分支梯度；短迭代验证其离开零点 |
| 可见性标签噪声过大 | 标签构造可视化；独立开关与消融 |
| 硬负样本误选真实目标 | GT 排除区、NMS 与样例可视化检查 |
| Copy-Paste 泄漏或产生不合理标注 | 只使用训练数据；检查几何、类别和框同步 |
| VDRM 与 CE token 删除不兼容 | 本机用合成输入检查 Block 6 时的动态搜索 token 数；服务器再验证真实训练 |
| 多卡参数未参与反向 | 本机不做结论；服务器 DDP smoke test 检查 unused parameters |
| 用测试集反复调参导致过拟合 | 完整训练前冻结超参数；统一阈值和协议 |
| 将 CTMP 收益错误归因给 VDRM | 始终保留四行主实验矩阵 |
| 本机无真实数据集 | 只验证合成前向/反向；不将 smoke test 当作效果结论 |
| 为适配本机误改服务器路径 | 路径文件保持不变；所有本机测试绕过真实 dataloader |

## 10. 实验记录规范

每个正式实验必须保存：

```text
experiment_id
purpose_or_hypothesis
code_version
config_file
dataset_versions_and_paths
pretrained_weight_and_checksum
random_seed
gpu_and_software_environment
start_time_and_end_time
checkpoint_paths
training_and_validation_logs
test_commands
raw_results
summary_table
conclusion
```

命名应能区分结构、损失和增强，不使用 `final`、`new`、`best2` 等无法追溯的名称。

## 11. 变更控制：防止后续偏移

### 11.1 不可静默改变的项目

以下任一变化都视为实验计划变更，必须先更新本文档并记录原因：

- 增删训练数据集或改变四数据集比例；
- 改变 300 epoch、epoch 240 降学习率或 60000 samples/epoch；
- 改变 MAE ViT-Base 初始化；
- 改变 CE 或 VDRM 插入位置；
- 改变主损失权重；
- 将 CTMP 或其他运动模块加入 VDRM 训练；
- 使用测试集反复选择结构或超参数；
- 改变统一 Tracker 接口的必需字段或语义；
- 为不同测试集使用不同 VDRM 权重或参数。

### 11.2 允许在主目标内探索的项目

以下项目允许消融，但必须一次只改变少量变量并完整记录：

- 部件原型数量与聚合方式；
- 部件—搜索 token 的匹配函数；
- 可靠性归一化与门控形式；
- 可见性标签构造；
- 排序损失 margin；
- GT 排除区与 NMS 半径；
- 结构化遮挡和同类 Copy-Paste 的概率与几何参数；
- `visual_reliability` 的汇聚形式。

这些探索不得改变 VDRM 的核心假设，也不得扩展为第二个推理或运动模块。

### 11.3 变更记录模板

```markdown
## Change-YYYYMMDD-NN

- Proposed change:
- Reason:
- Evidence:
- Affected experiments:
- Does it change a locked item: yes/no
- Required new baseline or ablation:
- Decision:
```

若没有对应记录，后续实现默认必须回到本文档锁定方案。

## 12. 最终交付物

本项目完成时应交付：

1. VDRM 网络实现与清晰注释；
2. 独立 VDRM 训练配置；
3. `L_visibility`、`L_rank` 和 warm-up 实现；
4. 第一版提供可开关的结构化遮挡；同类 Copy-Paste 作为后续可选消融；
5. 本机合成单元检查和 RTX 5070 单卡最小 smoke test；
6. 上传服务器后完成四卡 DDP smoke test，再执行 4× RTX 4090、300-epoch 完整训练；
7. 每 20 epoch checkpoint、验证结果与完整日志；
8. 五个统一测试集的结果和属性分析；
9. VDRM 最小消融表；
10. 与官方 OSTrack 兼容的 `TrackerOutput`；
11. 可直接交给 CTMP 项目替换 Backend 的权重、配置和接入说明。

---

**主线判断规则：** 如果一项改动不能直接服务于“可见性引导的部件表征”或“真实目标与硬干扰的判别学习”，则它不属于 VDRM 主线；若它涉及相机、轨迹、速度、搜索中心或框融合，则应放入 CTMP 项目。

**仓库上传规则：** 上传源码、配置、实验章程和测试脚本；不上传根目录 `data/`、`output/`、`pretrained_models/`、checkpoint、结果文件或模型权重。`lib/train/data/` 是源码目录，必须上传，不能用未锚定的 `data/` 忽略规则误排除。

## Change-20260726-01

- Proposed change: 将 VDRM-v1 的绝对 Top-4 相似度可靠性改为第一峰与空间去重后最强干扰峰的 margin；将 `L_rank` 从 Center Head 响应改为直接监督同一部件相似度图。
- Reason: V1 的可靠性衡量“是否存在模板相似内容”，无法可靠判断跟踪器选择的是否仍为原目标身份。
- Evidence: `uav_car15` 首次持续失败时 IoU 降至 0.083，但 VDRM 可靠性从失败前约 0.666 上升至约 0.706；`uav_car7` 在第 271 帧锁定错误同类目标时 VDRM 可靠性约 0.790、响应可靠性约 0.578，并持续失败到第 720 帧，而恢复正确时 VDRM 可靠性仅约 0.653。`uav_car12` 说明低置信度可检测部分遮挡失败，但不能解决上述同类身份混淆。
- Affected experiments: 新增 VDRM-v2 独立训练；V1 代码路径、配置和已有结果保留。
- Does it change a locked item: no。四数据集、比例、60000 samples/epoch、300 epoch、epoch 240 降学习率、CE 位置、Block 6 插入、损失权重、warm-up 和残差结构均不变。
- Required new baseline or ablation: 同协议比较重训 OSTrack、VDRM-v1、VDRM-v2；首先观察四个已登记 UAV123 序列，再统一测试五个数据集。不得依据单个测试集为 V2 设置专用阈值。
- Decision: approved。用户于 2026-07-26 授权开始修改和优化；仅实施可靠性与排序监督改动。

## Change-20260727-01

- Proposed change: 放弃 VDRM-v2，回退到 VDRM-v1；只增加训练期真实同类硬干扰 Copy-Paste。
- Reason: V2 的 margin 标定在训练中塌缩为近常数可靠性，并通过极低匹配尺度绕开了预期判别目标；继续修改 margin 或增加门控会扩大不可归因因素。
- Evidence: V2 最终 `alpha=-2.443376`、`scale=0.000169`、`bias=0.619119`，四个诊断序列的可靠性约固定为 `0.65`。在 `uav_car7` 第 271 帧错误锁定同类目标时，V2 仍给出接近最大匹配间隔，说明内部第一/第二峰 margin 不能识别目标身份。V2 相对重训 OSTrack 的 AUC 在 VisDrone、UAV123、UAVDT、DTB70、LaSOT 分别为 `-0.21/-1.31/-0.84/-0.07/-0.95`。
- Affected experiments: 新增 VDRM-v3-HNCP 独立训练；V1、V2 配置和结果只作复现与对照。
- Does it change a locked item: no。四训练数据集、比例、训练轮数、学习率、CE、VDRM 网络、残差、损失和 warm-up 均不变；只启用章程中预留的同类 Copy-Paste 训练增强。
- Required new baseline or ablation: 使用同一重训 OSTrack 和 VDRM-v1 对照 VDRM-v3-HNCP；先检查应用率及四个登记序列，再统一测试五个数据集。不得同时修改可靠性、残差或损失。
- Decision: approved。用户于 2026-07-27 明确要求下一版从 V1 回退并只做一个受控改动，禁止继续堆叠 margin、门控或额外质量头。

## Change-20260728-01

- Proposed change: 放弃将 V3-HNCP 作为下一版起点，再次回退到 VDRM-v1；只对 `alpha` 乘入后的完整搜索 token 残差施加相对输入 token 范数上限。
- Reason: V3 的真实同类 Copy-Paste 虽提高四个无人机数据集的平均 AUC，但在登记序列中产生了新的严重回归；继续调整 HNCP 概率或增加可靠性判断不能直接解决高置信度错误身份锁定。V1/V3 的自由标量最终均达到约 `alpha=-1.046`，当前残差没有幅度安全边界，错误匹配可以把完整模板原型更新注入后续骨干。
- Evidence: V3 相对 V1 在 `uav_car7` 的平均 IoU 从 `0.4236` 降至 `0.1594`，首次持续失败从第 271 帧提前到第 219 帧；`uav_car9` 从 `0.8612` 降至 `0.3700`，并从无持续失败变为第 800 帧后持续漂移。`uav_car7` 失败后的视觉可靠性约 `0.82–0.84`、第一响应峰约 `0.89`、第二峰比约 `0.001–0.003`，说明这是高置信度错误身份锁定，不是增加 margin 或置信度门控能够纠正的问题。
- Controlled variable: 新增固定 `RESIDUAL_MAX_RATIO=0.05`。设原搜索 token 为 \(f_i\)，未经限制的完整更新为 \(d_i=\alpha r_i\)，V4 使用
  \[
  d'_i=d_i\min\left(1,\frac{0.05\lVert f_i\rVert_2}{\lVert d_i\rVert_2+\epsilon}\right),
  \qquad f'_i=f_i+d'_i.
  \]
  上限参考范数停止梯度，避免骨干通过增大 token 范数绕开约束。该操作无新增可学习参数、无阈值分支、无质量头、无新损失；`alpha=0` 时仍严格保持 OSTrack 前向路径。
- Bound registration: `0.05` 在训练前通过合成整网探针固定。使用 V1 最终量级 `alpha=-1.046` 时，未经限制的平均更新约为输入 token 范数的 `0.0478`；`0.25/0.10/0.075` 均完全不介入，`0.03` 对全部 token 裁剪，`0.05` 仅裁剪约 `34.4%` 的高幅度尾部并将平均更新从 `0.0478` 降至 `0.0463`。因此选择 `0.05` 作为“限制异常尾部而不压平主体更新”的一次性预注册值，不依据任何测试集结果调整。
- Compatibility: 默认 `RESIDUAL_MAX_RATIO=0.0` 表示关闭约束，V1、V2、V3 的配置、checkpoint 与前向结果保持兼容。V4 明确关闭 HNCP，网络、Top-4 可靠性、结构化遮挡、两个辅助损失、warm-up、训练数据和完整训练协议均回到 V1。
- Observability: 训练日志和逐帧诊断新增 `residual_clip_rate`、`raw_delta_relative_norm`、`delta_relative_norm`。这些字段只用于判断约束是否介入，不参与前向决策或损失。
- Required comparison: 同协议比较重训 OSTrack、VDRM-v1、VDRM-v3-HNCP、VDRM-v4-BR。完整测试前先检查 `uav_car7`、`uav_car9`、`uav_car12`、`uav_car15`；V4 至少不得重现 V3 在 car7/car9 上的新回归，同时观察 car15 是否改善。
- Decision: approved。用户于 2026-07-28 要求在 CTMP 尚未完成时继续优化一个版本的模块一；本次仅实施残差范数边界。

## 13. 第一版实现状态与执行入口

截至第一版实现，已完成：

- 固定 `2×2` 部件原型 VDRM；
- Block 6 后插入与零初始化残差；
- 部件可靠性和 `visual_reliability` 输出；
- 搜索目标区域结构化遮挡及软可见性标签；
- `L_visibility` 与无 margin `L_rank`；
- 前 20 epoch 辅助损失 warm-up；
- VDRM 独立 `4e-4` 优化器参数组；
- 独立 300-epoch 配置；
- 无数据集 CPU/单张 RTX 5070 smoke test。

第一版配置：

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_32x4_ep300.yaml
```

本机测试命令：

```powershell
D:\Anaconda\envs\track\python.exe -m unittest discover -s tests -v
D:\Anaconda\envs\track\python.exe tracking\smoke_test_vdrm.py --device cpu
D:\Anaconda\envs\track\python.exe tracking\smoke_test_vdrm.py --device cuda
```

本机测试只验证代码连通性、张量形状、损失、梯度、参数学习率和最小 checkpoint 恢复，不产生效果结论。

上传服务器后必须先执行：

1. 检查服务器数据路径仍为原值；
2. 检查 MAE 权重可以加载；
3. 单卡真实 batch 前向/反向；
4. 四卡 DDP 数十个 iteration；
5. 四卡 1 epoch smoke test；
6. 检查日志、梯度、显存、checkpoint 恢复和验证流程；
7. 以上全部通过后才能启动 300 epoch。

仓库初始化或上传前检查：

```powershell
git status --short
git check-ignore -v data output pretrained_models
git check-ignore -v lib/train/data/processing.py
```

预期前三个根目录被忽略，而 `lib/train/data/processing.py` 和 `lib/train/data/vdrm_augmentation.py` 不应被忽略。

## 14. VDRM-v3-HNCP 实现状态与执行入口

VDRM-v3-HNCP 只增加训练期真实同类干扰数据增强：

- V1 与 V3 的 `MODEL`、`TRAIN` 配置及同随机种子初始化参数已验证完全一致；
- 同类实例必须来自同一训练数据集的类别索引且不是当前实例；
- 粘贴失败、类别未知或无另一实例时安全跳过；
- 真实目标区域与标注不被覆盖；
- 验证集和推理不采样、不粘贴干扰；
- 不新增可学习参数、margin、loss、门控或质量头。

本机已执行单元测试、真实 COCO 样本链路、batch collate、CPU smoke 和单张 RTX 5070 smoke。真实训练数据路径仍以服务器现有配置为准，本机检查未改写任何路径。

服务器运行顺序：

```bash
python -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES=0 \
python tracking/smoke_test_vdrm.py \
  --config experiments/ostrack/vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300.yaml \
  --device cuda

python tracking/smoke_test_vdrm_data.py \
  --config vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300 \
  --batches 4 \
  --batch-size 8 \
  --workers 2 \
  --use-lmdb 0
```

上述三项通过后，才启动完整训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

训练日志中检查 `VDRM/distractor_applied_rate` 是否长期大于 0 且大致围绕配置概率波动。该统计只验证数据增强确实生效，不能单独作为效果结论。

## 15. VDRM-v4-BR 设计与执行入口

VDRM-v4-BR（Bounded Residual）从 V1 配置出发，只限制 VDRM 对每个搜索 token 的完整残差更新幅度：

- `RESIDUAL_MAX_RATIO: 0.05`；
- HNCP 明确关闭，`VDRM_DISTRACTOR_PROBABILITY: 0.0`；
- 不改变 Top-4 可靠性、部件原型、残差方向、插入层、损失或训练协议；
- 不增加参数、margin、门控、质量头或推理阈值；
- V1/V2/V3 默认使用 `RESIDUAL_MAX_RATIO: 0.0`，保持旧实验行为。

配置：

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_v4_br_32x4_ep300.yaml
```

服务器首先执行：

```bash
python -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES=0 \
python tracking/smoke_test_vdrm.py \
  --config experiments/ostrack/vitb_256_mae_ce_vdrm_v4_br_32x4_ep300.yaml \
  --device cuda
```

只有上述检查和短 DDP 检查通过后，才启动完整训练。训练日志必须同时保留：

- `VDRM/alpha`；
- `VDRM/residual_clip_rate`；
- `VDRM/raw_delta_relative_norm`；
- `VDRM/delta_relative_norm`。

若 `residual_clip_rate` 长期接近 0，说明上限没有实际介入，V4 不能用于证明残差边界有效；若长期接近 1，则需要结合主损失和验证结果判断是否约束过强，不允许只凭裁剪率修改上限。

## Change-20260729-01

- Proposed change: 放弃 V4 作为下一版起点，完整回到 V3-HNCP；只把 HNCP 实际粘贴框传给 `L_rank`，使已应用 HNCP 的样本使用粘贴框内最高 Center Head 响应作为硬负样本。未应用 HNCP 的样本继续使用全背景最高响应。
- Reason: V3 已加入真实同类干扰，但原增强函数只返回是否粘贴，没有返回粘贴坐标；`L_rank` 因此仍在整个背景中盲选最高响应，训练监督未必作用于刚加入的同类干扰。
- Evidence: V4 相对 V3 的 AUC 在 VisDrone、UAV123、UAVDT、DTB70、LaSOT 上分别变化 `+0.18/-0.57/-2.25/+0.63/-0.10`，不能替代 V3 主线。代码审计确认 V3 的 `vdrm_distractor_applied` 仅用于日志，而原排序负样本由 `gaussian_map <= 0` 的全部位置直接取最大值，HNCP 位置没有进入损失选择。
- Controlled variable: 新增固定布尔开关 `TRAIN.VDRM_ALIGN_DISTRACTOR_RANK=True`。排序形式仍为 `softplus(l_neg-l_pos)`，权重仍为 `0.5`；只将 HNCP 已应用样本的 `l_neg` 选取区域从“全部背景”改为“粘贴框与背景的交集”。极小干扰框没有覆盖响应图单元中心时，使用距离干扰框中心最近的背景单元，避免静默丢弃监督。
- Compatibility: 默认开关为 `False`，V1/V2/V3/V4 的已有配置仍保持原全背景硬负样本行为。V5 明确使用 V3 的网络、Top-4 可靠性、无边界残差、HNCP 概率、数据、损失、warm-up 和训练协议。
- Observability: 新增 `VDRM/alignment_success_rate` 和 `VDRM/distractor_rank_margin`，只用于检查坐标传递和排序难度，不参与前向融合、门控或损失加权。
- Does it change a locked item: no。四数据集、1:1:1:1、60000 samples/epoch、300 epoch、epoch 240 降学习率、学习率、CE 位置、Block 6 插入、损失权重和 checkpoint 策略均不变。
- Required new baseline or ablation: 同协议比较重训 OSTrack、VDRM-v3-HNCP 与 VDRM-v5-AHNCP；先检查四个登记序列，再统一测试五个数据集。不得同时修改 HNCP 概率、可靠性、残差、margin、门控或质量头。
- Decision: approved。用户于 2026-07-29 明确要求下一版回到 V3，只修正 HNCP 干扰位置与排序监督不对齐。

## 16. VDRM-v5-AHNCP 设计与执行入口

V5 是 V3 的受控监督对齐版本，不增加网络结构或可学习参数：

1. HNCP 完成粘贴后返回搜索图坐标系中的归一化 `xywh` 干扰框；
2. 数据处理将干扰框与 `vdrm_distractor_applied` 一同传给 Actor；
3. 对已应用 HNCP 的样本，在粘贴框内取最高背景响应 `l_neg`；
4. 对未应用 HNCP 或元数据无效的样本，保持 V3 的全背景最高响应回退；
5. 正样本响应、排序函数、辅助损失权重及 warm-up 均不改变。

配置：

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_v5_ahncp_32x4_ep300.yaml
```

训练前必须观察：

- `VDRM/distractor_applied_rate`：实际应用 HNCP 的 batch 比例；
- `VDRM/alignment_success_rate`：已应用 HNCP 的样本中成功找到对齐负样本的比例，正常应接近 `1.0`；
- `VDRM/distractor_rank_margin`：真实目标峰减去粘贴干扰峰，允许训练早期为负，但应结合 `Loss/vdrm_rank` 观察其趋势；
- `VDRM/alpha` 与 V3 的演化是否同量级，防止无关结构变化。

上述诊断只判断实现和优化过程，不能代替完整数据集效果对比。V5 不允许根据单个测试数据集另设阈值或参数。

## Change-20260730-01

- Proposed change: 放弃 V5-AHNCP 作为下一版起点，完整回到 V3-HNCP；只把 HNCP 的粘贴位置由“24 次候选中的首个有效位置”改为“同一组 24 次候选中距离真实目标最近的有效位置”。
- Reason: V5 虽然修正了粘贴位置与排序监督的坐标对齐，但训练后粘贴干扰已变成过于容易的负样本，削弱了 V3 原有的全背景最强负样本监督。下一步应提高 HNCP 数据本身的困难度，而不是继续改变损失、门控、可靠性或残差结构。
- Evidence: V5 末期目标与粘贴干扰的响应差约为 `12.77`，且 V5/V3 的排序损失比值约为 `0.144/0.182=0.791`，与未应用 HNCP 的样本比例 `1-0.208=0.792` 基本一致。这说明 V5 中已应用 HNCP 的样本几乎不再贡献排序梯度。V5 相对重训 OSTrack 的 AUC 在 UAV123、DTB70 分别为 `+0.15`、`+0.45`，但仍未同时达到预设的 `+0.3~0.5` 改善目标。
- Controlled variable: 固定 `DATA.SEARCH.VDRM_DISTRACTOR_PLACEMENT=nearest`。对每个满足“不覆盖真实目标且不进入无效填充区”的候选框，计算
  \[
  d^2=\left(\frac{x_d-x_t}{w_t+w_d}\right)^2+\left(\frac{y_d-y_t}{h_t+h_d}\right)^2,
  \]
  并选择 `d²` 最小者；候选次数、尺寸范围、应用概率及所有有效性条件保持 V3 不变。
- Loss and network: 使用 V3 的全背景最高响应 `L_rank`，明确设置 `TRAIN.VDRM_ALIGN_DISTRACTOR_RANK=False`；VDRM 网络、Top-4 可靠性、零初始化残差、两个辅助损失及其 warm-up 均不改变。
- Observability: 新增只读日志 `VDRM/distractor_hard_hit_rate` 与 `VDRM/distractor_global_gap`。前者表示全局最强背景响应是否落在粘贴框内，后者表示全局最强背景响应减去粘贴框内最强响应；两者不参与前向决策、损失或加权。
- Compatibility: 默认放置模式仍为 `random`，默认困难度日志关闭；V1 至 V5 的已有配置、checkpoint 和训练行为不变。
- Does it change a locked item: no。四训练数据集及比例、60000 samples/epoch、300 epoch、epoch 240 降学习率、学习率、CE 位置、Block 6 插入、batch size、损失权重和 checkpoint 策略全部不变。
- Required comparison: 同协议比较重训 OSTrack、V3-HNCP、V5-AHNCP 和 V6-Near-HNCP。预注册成功标准为 UAV123 AUC 至少 `68.57`、DTB70 AUC 至少 `66.86`，即两者均比重训 OSTrack 至少提高 `0.30`；不得根据单个测试集修改参数。
- Decision: approved。用户于 2026-07-30 明确要求实施“V3 + 近目标 HNCP”。

## 17. VDRM-v6-Near-HNCP 设计与执行入口

V6 是 V3 的单变量数据增强版本：

1. HNCP 来源、概率、尺寸范围、候选次数与 V3 完全一致；
2. 仍禁止粘贴框覆盖真实目标或落入搜索图无效填充区；
3. 只在已有有效候选中选择离真实目标最近的位置；
4. 排序损失仍选取全背景最高错误响应，不使用 V5 的粘贴框对齐负样本；
5. 不增加可学习参数、margin、门控、质量头、可靠性分支或推理逻辑。

配置：

```text
experiments/ostrack/vitb_256_mae_ce_vdrm_v6_near_hncp_32x4_ep300.yaml
```

固定随机种子的 100 组几何探针中，近目标选择将归一化中心距离均值从 `0.7295` 降至 `0.5482`，下降 `24.9%`，且 100 组均不比原首个有效候选更远。该结果只证明放置策略按设计生效，不代表跟踪精度一定提高。

训练日志必须保留：

- `VDRM/distractor_applied_rate`；
- `VDRM/distractor_hard_hit_rate`；
- `VDRM/distractor_global_gap`；
- `Loss/vdrm_rank`、`VDRM/alpha` 与 `VDRM/reliability`。

`distractor_hard_hit_rate` 越高、`distractor_global_gap` 越低，只表示粘贴干扰更可能成为全局硬负样本，不能代替五数据集最终测试。V6 不输出 V5 的 `alignment_success_rate`，因为本版明确恢复 V3 的全局负样本排序。
