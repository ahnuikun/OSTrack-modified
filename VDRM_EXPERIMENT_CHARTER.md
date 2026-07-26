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
