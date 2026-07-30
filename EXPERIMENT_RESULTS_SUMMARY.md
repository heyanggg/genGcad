# SmartGen_GCAD 主实验结果与补充实验建议

更新日期：2026-07-30

## 1. 当前结论

- 当前约定的主实验为 3 个数据集（FR、SP、US）× 3 个迁移任务（winter→spring、daytime→night、single→multiple）× 单 seed（2024），共 9 组；这9组均属于完整方案（GCAD-on），其中两组因可靠性门控没有可注入关系而自动退化。
- 9 组正式结果均已完成，正式清单、结果文件和校验信息齐全；从“完成当前单 seed 主实验矩阵”的角度看，主实验已符合要求。
- 9 组宏平均 F1 为 **0.955077**，宏平均 Accuracy 为 **0.953864**。
- 旧的 3 组 GCAD-off 都复用了 on 归档的 SSC 文件，不符合本项目现在采用的定义“off 必须完全跳过 GCAD 并独立执行原 SmartGen 的 SSC 路径”。这些旧 off 及其多 seed 分析已全部降为 `diagnostic`，不得再作为正式消融结论。
- SP single→multiple 已按正确定义重跑：清单明确记录 `pipeline_mode=original_smartgen_without_gcad`、`gcad.module_executed=false`、`compression.training_mode=trained`。单 seed F1 为 **0.908046**，对应 on 为 **0.913295**。
- 正确 SP off 的5折三检测器 seed结果仍显示显著划分敏感性：单次80/20检测 F1 为0.9080，但5折稳健 F1为0；其连续分数 AUROC为 **0.4063±0.2036**，on为 **0.8866±0.0678**。因此既不能用旧的极端 `0 vs 0.913`，也不能只用新单次的 `0.908 vs 0.913` 下结论。

## 2. 正式主实验汇总

| 数据集 | 迁移任务 | F1 | Accuracy | Precision | Recall | TOF 数量（输入→过滤后→最终） | GCAD 关系（发现→注入） |
|---|---|---:|---:|---:|---:|---:|---:|
| FR | winter→spring | 0.961749 | 0.960227 | 0.926316 | 1.000000 | 196→172→187 | 5→5 |
| FR | daytime→night | 0.971924 | 0.971113 | 0.945382 | 1.000000 | 28→28→28 | 未启用 |
| FR | single→multiple | 0.994475 | 0.994444 | 0.989011 | 1.000000 | 111→94→107 | 1→1 |
| SP | winter→spring | 0.971122 | 0.970467 | 0.950066 | 0.993132 | 216→193→205 | 2→2 |
| SP | daytime→night | 0.984420 | 0.984174 | 0.969318 | 1.000000 | 52→48→51 | 未启用 |
| SP | single→multiple | 0.913295 | 0.905063 | 0.840426 | 1.000000 | 88→72→83 | 1→1 |
| US | winter→spring | 0.953375 | 0.952586 | 0.937781 | 0.969496 | 89→80→84 | 44→12 |
| US | daytime→night | 0.903587 | 0.908556 | 0.955506 | 0.857020 | 72→62→68 | 19→11 |
| US | single→multiple | 0.941746 | 0.938143 | 0.889906 | 1.000000 | 297→256→297 | 16→16 |

### 混淆矩阵

| 数据集 | 迁移任务 | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|
| FR | winter→spring | 88 | 81 | 7 | 0 |
| FR | daytime→night | 952 | 897 | 55 | 0 |
| FR | single→multiple | 90 | 89 | 1 | 0 |
| SP | winter→spring | 723 | 690 | 38 | 5 |
| SP | daytime→night | 1706 | 1652 | 54 | 0 |
| SP | single→multiple | 158 | 128 | 30 | 0 |
| US | winter→spring | 2924 | 2822 | 194 | 92 |
| US | daytime→night | 2985 | 3344 | 139 | 498 |
| US | single→multiple | 3209 | 2812 | 397 | 0 |

### 分数据集平均 F1

| 数据集 | 三组平均 F1 |
|---|---:|
| FR | 0.976049 |
| SP | 0.956279 |
| US | 0.932903 |

## 3. “怎么证明 GCAD 有效”的准确回答

可以直接这样回答：

> 当前完整方案优于历史 Gen，能证明 SmartGen_GCAD 整体方案有效；但不能仅凭这一点把增益归因给 GCAD。此前三组 off 复用了 on 的 SSC 归档，不符合现采用的严格 off 定义，已经撤销正式资格。正确 SP off 的单次 F1 与 on 接近，但5折连续分数明显弱于 on，说明存在正向信号，同时也存在很强的生成与检测划分敏感性。应先按同一定义重跑 US 两组 off，并补充独立 LLM 生成重复，再用 shuffled-GCAD 排除“只是多了一段提示词”的解释。

“比 Gen 高”回答的是：

```text
完整新系统是否比旧系统好？
```

师兄反复追问的其实是：

```text
在完整新系统内部，去掉 GCAD 后是否变差？
真实 GCAD 是否比随机关系更好？
```

这是两个不同的问题。前者是总体比较，后者是组件归因。

## 4. GCAD 原论文如何证明其有效

参考论文：Zehao Liu、Mengzhou Gao、Pengfei Jiao，**GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality**，AAAI 2025。可查看 [arXiv 页面](https://arxiv.org/abs/2501.13493) 和 [AAAI 正式论文 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/34096/36251)。

原论文不是只靠“总结果更高”来证明，而是形成了四层证据链：

1. **总体性能比较**：在 SWaT、SMD、MSL、SMAP、PSM 五个真实数据集上，与 DAGMM、USAD、GDN、Anomaly Transformer、GANF、MEMTO 等基线比较，说明完整 GCAD 具有竞争力。
2. **组件消融**：分别移除 Granger causality、temporal correlations 和 graph sparsification。论文 Table 3 中，完整 GCAD 在 SWaT 上 ROC/PRC 为 0.8690/0.7758，在 SMD 上为 0.9533/0.7502；去掉 Granger causality 后分别降为 0.8478/0.7513 和 0.9405/0.6424。这一部分才是对“GC 模块有效”的直接证据。
3. **参数敏感性**：考察最大时间滞后和稀疏化阈值，说明效果不是只来自偶然选中的一个参数。
4. **案例解释**：在 SWaT 攻击案例中展示因果偏差矩阵，并把异常关系与实际物理攻击链对应，证明检测结果具有机制层面的可解释性。

| 证据 | GCAD 原论文 | 当前项目应采用的对应证据 |
|---|---|---|
| 总体效果 | 与多个异常检测基线比较 | SmartGen_GCAD 与原 Gen 比较 |
| 组件归因 | 去掉 GC、TC、稀疏化的消融 | GCAD-on 与 GCAD-off；再加入 shuffled-GCAD |
| 稳定性 | 时间滞后、阈值敏感性 | 多次重复；关系数上限、history、稀疏阈值敏感性 |
| 可解释性 | 物理攻击链案例 | 展示生成序列是否遵守发现的源→目标及滞后关系 |

## 5. 当前项目与原论文的边界

当前项目采用的是 **GCAD-derived lagged directional dependencies**：先学习带滞后的方向关系，再将筛选后的关系作为软约束注入 SmartGen 提示词。它复用了原论文“预测梯度→动态 Granger 关系→稀疏化”的思路，但没有照搬原论文最后的“正常因果图与测试因果图偏差评分”作为异常分数。

因此论文表述应为：

> 本项目将 GCAD 派生的滞后方向依赖用于指导 SmartGen 的跨环境数据生成。

不宜表述为：

> 本项目完整复现了原 GCAD 异常检测器。

原论文可以提供论证方法和理论动机，但当前项目仍必须通过自身的 on/off 消融来证明 GCAD 引导对生成和下游检测有效。

## 6. GCAD 消融结果与后续实验

### 6.1 旧 off 结果：定义不符合要求，仅作历史诊断

三组均复用了对应正式 on 组归档内的 SSC 选择文件。虽然当前流水线中 GCAD 只读取 `split_trn.pkl`、并不会改写 SSC，复用 SSC 可构成一种“只改变提示词”的受控消融；但它不符合本项目现在明确采用的定义：**GCAD-off 必须完全不执行 GCAD，并独立运行原 SmartGen 的 SSC 阶段**。因此下表不再是正式 on/off 结果。

| 任务 | GCAD-on F1 | GCAD-off F1 | on−off | on TOF | off TOF |
|---|---:|---:|---:|---:|---:|
| US winter→spring | 0.953375 | 0.937166 | +0.016208 | 89→80→84 | 94→83→90 |
| US daytime→night | 0.903587 | 1.000000 | -0.096413 | 72→62→68 | 72→63→63 |
| SP single→multiple | 0.913295 | 0.000000 | +0.913295 | 88→72→83 | 115→98→110 |

审计结论：

- US night 检测器只使用小时列，正常测试样本只含小时编码 0/1/6/7，攻击只含 2/3/4/5，攻击本身接近可分，因此单次出现满分并非代码伪造。
- SP multiple 检测器只使用设备列；原协议还将同一份生成数据同时用于训练和阈值验证。off 生成数据更频繁复现攻击中的 Television 设备模式，使攻击分数低于正常分数，因而得到 F1=0。
- 两组极端数值都有可解释原因，但样本量、验证集和单 seed 不足以支持组件归因；三组 F1 宏平均也会被 SP 的极端差值主导，不能继续使用。

这些归档仍完整保留模型、SSC、prompt/response、原始与解析生成数据、TOF 中间产物、检测数据和 checksum，但已从 `formal` 移至 `diagnostic`。

### 6.2 旧 off 的多 seed 检测：随旧定义一起降为诊断性

该分析统一采用：

- on/off 取相同生成序列数量；
- 每个模式重新按 80%/20% 划分互不重叠的训练集与阈值验证集；
- 检测器 seed 为 2024、2025、2026；
- 阈值分位数仍与正式任务一致；
- 同时报告 F1、AUROC 和 AUPRC。

以下为三检测器 seed 的均值 ± 样本标准差：

| 任务 | 模式 | F1 | AUROC | AUPRC |
|---|---|---:|---:|---:|
| US winter→spring | on | 0.9631 ± 0.0061 | 0.9693 ± 0.0044 | 0.9442 ± 0.0165 |
| US winter→spring | off | 0.9500 ± 0.0303 | 0.9874 ± 0.0013 | 0.9783 ± 0.0026 |
| US daytime→night | on | 0.9444 ± 0.0485 | 0.9986 ± 0.0020 | 0.9986 ± 0.0020 |
| US daytime→night | off | 0.9450 ± 0.0934 | 0.9901 ± 0.0171 | 0.9918 ± 0.0142 |
| SP single→multiple | on | 0.6115 ± 0.5296 | 0.9080 ± 0.0550 | 0.8741 ± 0.0641 |
| SP single→multiple | off | 0.0000 ± 0.0000 | 0.2337 ± 0.2708 | 0.5545 ± 0.0986 |

配对的 on−off 均值：

| 任务 | ΔF1 | ΔAUROC | ΔAUPRC | 结论 |
|---|---:|---:|---:|---|
| US winter→spring | +0.0132 | -0.0181 | -0.0341 | 指标方向冲突，不能判为稳定获益 |
| US daytime→night | -0.0006 | +0.0085 | +0.0068 | F1 实质持平；任务过易 |
| SP single→multiple | +0.6115 | +0.6744 | +0.3197 | 连续分数明显支持 GCAD-on，但阈值 F1 仍不稳定 |

SP on 的三个 F1 为约 0.916、0、0.919，而对应 AUROC 均约为 0.875–0.972。这直接证明 F1=0 可能由小验证集估计出的阈值造成，论文中应将 AUROC/AUPRC 与 F1 并列，不能只报最好或单次 F1。

完整结果仍位于 `experiment_archive/analysis/gcad_off_balanced/`，包括 18 个检测器模型、逐 seed 数据划分、连续异常分数、指标和 SHA-256 校验文件；由于 off 输入定义已作废，这些结果只用于追溯异常，不用于论文消融结论。

### 6.3 已完成：按严格定义重跑 SP single→multiple off

代码现已硬性保证：

- `gcad-mode=off` 时不提取、不读取任何 GCAD 产物；
- off 与 `--reuse-sppc-selection-dir` 同时出现时直接报错；
- off 独立执行 Split、Dayse、SSC训练与选择、原始提示词生成、TOF和下游异常检测；
- 实验清单记录 `gcad.module_executed=false` 和 `compression.training_mode=trained`。

独立重训产生的30个 SSC 文件与 on 归档逐文件 SHA-256 完全一致（30/30）。这不是再次复用，而是因为 GCAD 只读、固定 seed 的 SSC 对同一输入产生确定性结果；两侧运行路径已经独立。

正确 off 共解析116条候选，TOF最终保留110条。正式单 seed结果为：

| 模式 | F1 | Accuracy | Precision | Recall | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| on | 0.913295 | 0.905063 | 0.840426 | 1.000000 | 158 | 128 | 30 | 0 |
| off（独立SSC） | 0.908046 | 0.898734 | 0.831579 | 1.000000 | 158 | 126 | 32 | 0 |

单次 on−off F1 为 **+0.005249**，只能说明这一次 on 略高，不能单独证明稳定收益。

随后采用5折交叉训练，确保每条校准分数来自未训练过该序列的模型。83条校准数据不足以稳定估计99分位：5折后 on/off 的99分位 F1 都为0。因此保留99分位结果作为协议记录，同时增加不使用攻击标签的固定稳健阈值。每一折只用该折验证分数计算中心和尺度，先独立标准化该折的测试分数，再对5折标准化分数取均值：

```text
z_fold = (score - median(validation_fold)) / (1.4826 × MAD(validation_fold))
anomaly if mean(z_fold) >= 3.5
```

严格 on/off 的5折三检测器 seed结果：

| 模式 | 稳健 F1 | AUROC | AUPRC |
|---|---:|---:|---:|
| on | 0.7904 ± 0.0207 | 0.8866 ± 0.0678 | 0.8161 ± 0.0794 |
| off（独立SSC） | 0.0000 ± 0.0000 | 0.4063 ± 0.2036 | 0.5922 ± 0.1030 |

单次80/20检测的0.908与5折结果的0之间存在明显冲突，说明 SP 的生成数据和阈值对训练划分非常敏感。论文不能只选其中一个数；应同时报告独立生成重复、AUROC/AUPRC和完整混淆矩阵。

正确结果位于：

- 正式单 seed：`experiment_archive/formal/sp_single_to_multiple_SPPC_th-0.915_gpt-5.6-sol__seed2024_ablation_gcad_off_fresh_ssc_original__detector-p99-seed2024/`
- 多 seed 5折：`experiment_archive/analysis/gcad_off_crossfit_fresh_ssc/`

### 6.4 历史原 Gen 为什么不为0

历史原 Gen 的 SP multiple 最终数据共89条，生成后端为 `gpt-4o-2024-11-20`，使用 OpenAI API 的 `temperature=0`、`top_p=0`、`seed=2024` 和 `max_tokens=8040`。当前严格 off 虽然走原 SmartGen 模块路径，但生成模型为 GPT-5.6 Sol，且通过 Codex CLI 调用，不能应用上述四个采样字段，所以两者不是同一份生成基线。

将历史原 Gen 数据放入与当前实验完全相同的5折、三检测器 seed协议后：

| 数据 | 稳健 F1 | AUROC | AUPRC |
|---|---:|---:|---:|
| 当前 GCAD-on | 0.7904 ± 0.0207 | 0.8866 ± 0.0678 | 0.8161 ± 0.0794 |
| 历史原 Gen（GPT-4o） | 0.7242 ± 0.0252 | 0.4534 ± 0.0709 | 0.5825 ± 0.0481 |
| 当前严格 off（GPT-5.6） | 0.0000 ± 0.0000 | 0.4063 ± 0.2036 | 0.5922 ± 0.1030 |

历史原 Gen 的三个 seed 都检出全部158条攻击，但同时产生125、133、104个误报。其 F1 不为0主要是因为模型倾向于把大量样本都判为异常，并不表示它能可靠区分正常与攻击；接近随机的 AUROC 更能揭示这一点。当前严格 off 与历史原 Gen 的连续排序能力都较弱，差距没有旧单次 F1 表面上那么大。

完整结果位于 `experiment_archive/analysis/gcad_on_vs_historical_original_gen_crossfit/`。

### 6.5 采样参数如何真正应用

当前 Codex CLI 只实际应用模型和 `reasoning_effort=none`，不接受 `temperature`、`top_p`、请求级 `seed` 或 `max_tokens`。实验清单已经把这四项标记为“原 Gen 请求值，但未应用”，不再伪装成生效参数。

要真正应用这些参数，需要增加 OpenAI API 后端并使用具备对应模型权限的 API key。API层可以设置 `temperature`、`top_p` 和输出 token 上限；Chat Completions 的 `seed` 只是 best-effort，并不保证完全确定。当前环境没有 `OPENAI_API_KEY`，因此还不能实际运行和验证该后端。即使传输方式本身不应改变实验逻辑，生成模型、采样字段是否生效会直接影响复现性。

### 6.6 下一优先项：其余严格 off 与 LLM 独立生成重复

SP off 已按严格定义重跑；US spring、US night 的旧 off 已降为诊断性，必须先按严格定义重跑。随后 on/off 两侧仍需按预先固定次数补充独立生成，不能把检测器 seed 当作生成重复。

当前生成后端不保证完全可由 seed 决定，因此应称为“独立重复运行”，不能暗示 LLM 输出严格确定性。需报告每个任务的 on−off 分布、均值 ± 标准差，以及 bootstrap 置信区间或配对检验。

### 6.7 后置项：随机关系负对照

至少在 US winter→spring 上加入 `shuffled-GCAD`：

- 保持与真实 GCAD 完全相同的关系条数、权重格式和提示词长度；
- 打乱源节点、目标节点或 lag；
- 比较 `real GCAD > shuffled GCAD ≈ GCAD-off` 是否成立。

这个对照相当于安慰剂：它检验真实关系的语义是否有效，而不是仅仅因为提示词变长或多了若干格式化关系。它不是为了保证 shuffled 结果更差；若 `real GCAD` 没有稳定优于 shuffled，就不能把提升归因给关系正确性。

本轮没有启动 shuffled，因为 on/off 尚未经过生成重复，且 US spring 的 AUROC/AUPRC 方向与 F1 相反。应先稳定 on/off，再选择确有正向信号的任务做 shuffled。

### 6.8 机制证据

除下游 F1 外，建议报告：

- **关系遵守率**：生成序列中，GCAD 的 source→target 是否在预测 lag 范围内出现；
- **分布接近度**：目标环境与生成数据在时间、设备、动作、序列长度上的 TV/JS 距离；
- **生成有效率**：原始生成数、解析成功数、TOF 后保留数、最终可用数；
- **关系稳定性**：不同 GCAD 内部 seed 下重复出现的边比例；
- **预测基线门控**：GCAD 预测器相对频率基线是否确有提升。

理想证据链为：

```text
GCAD 找到稳定且非平凡的关系
        ↓
真实关系确实改变生成序列，且比随机关系更符合目标环境
        ↓
下游异常检测结果稳定优于 GCAD-off
```

## 7. 主实验现状中的限制

- 当前正式矩阵是单 seed/单次生成，能够作为主结果，但不足以估计生成波动。
- US night、US spring 的旧 off 不符合严格定义，必须重跑后才能恢复正式 on/off 结论。
- `experiment-seed=2024` 控制 SSC、TOF、GCAD holdout 和异常检测器，但 Codex CLI 不应用 temperature、top_p、generation seed 和 max_tokens；因此相同调用次数也会产生不同数量和内容的序列。
- FR daytime→night、SP daytime→night 因门控未启用 GCAD，只能说明系统在无可靠关系时可安全退化，不能作为“GCAD 带来提升”的证据。
- FR winter→spring 缺少可选的 TOF candidate scratch 中间文件，但正式复现所需的核心产物、指标与清单完整。

## 8. 推荐执行顺序

1. SP multiple 的严格 off、正式单 seed、5折三检测器 seed与历史原 Gen同协议分析均已完成。
2. 下一步按严格定义重跑 US winter→spring off，再重跑 US daytime→night off。
3. 为 on/off 补充预先固定的独立 LLM 生成重复。
4. 若真实关系在重复后仍有正向信号，再做 shuffled-GCAD 强负对照。
4. 生成重复后，优先选择持续出现正向信号的任务做 shuffled-GCAD；不预先假定一定是 US spring。
5. 同步计算关系遵守率，证明 GCAD 提示确实改变了生成序列中的滞后关系。
6. 最后再考虑 history、关系上限或稀疏阈值等参数敏感性。

因此，主实验无需推倒重跑；GCAD-off 的检测评估问题已经纠正。当前最重要的是增加独立生成重复，而不是先做 shuffled，也不是人为消除 0 或 1。
