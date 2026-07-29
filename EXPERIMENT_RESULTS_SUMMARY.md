# SmartGen_GCAD 主实验结果与补充实验建议

更新日期：2026-07-30

## 1. 当前结论

- 当前约定的主实验为 3 个数据集（FR、SP、US）× 3 个迁移任务（winter→spring、daytime→night、single→multiple）× 单 seed（2024），共 9 组。
- 9 组正式结果均已完成，正式清单、结果文件和校验信息齐全；从“完成当前单 seed 主实验矩阵”的角度看，主实验已符合要求。
- 9 组宏平均 F1 为 **0.955077**，宏平均 Accuracy 为 **0.953864**。
- 3 组首轮 GCAD-on/off 受控消融也已完成：两组 on 优于 off，一组 night 任务出现反例。三组 on 的平均 F1 为 **0.923419**，off 为 **0.645722**，但差异主要由 SP single→multiple 贡献，且 Codex 生成并不受 seed 严格控制，暂时不能据此声称 GCAD 在所有任务上稳定有效。

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

> 当前完整方案优于 Gen，能证明 SmartGen_GCAD 整体方案有效；但不能仅凭这一点把增益归因给 GCAD。我们已经固定数据划分、SSC 选择、提示词主体、生成模型、生成类别与调用次数、TOF、异常检测器和评估方法，仅关闭 GCAD 关系段进行了首轮消融。结果是 US spring 和 SP multiple 支持 GCAD，US night 不支持，说明 GCAD 的效果具有任务依赖性。下一步要通过多次独立重复估计 LLM 生成波动；若 GCAD-on 的均值在多个任务上稳定优于 GCAD-off，并且真实 GCAD 又优于同边数的随机或打乱关系，才能严谨地把提升归因给 GCAD。

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

### 6.1 已完成：首轮干净的 GCAD-off 消融

三组均复用了对应正式 on 组归档内的 SSC 选择文件。逐类别字节检查表明，off 提示词相对 on 提示词只删除了 GCAD guidance 段，其余提示内容一致。

| 任务 | GCAD-on F1 | GCAD-off F1 | on−off | on TOF | off TOF | 首轮结论 |
|---|---:|---:|---:|---:|---:|---|
| US winter→spring | 0.953375 | 0.937166 | +0.016208 | 89→80→84 | 94→83→90 | 小幅支持 GCAD |
| US daytime→night | 0.903587 | 1.000000 | -0.096413 | 72→62→68 | 72→63→63 | 反例，off 更好 |
| SP single→multiple | 0.913295 | 0.000000 | +0.913295 | 88→72→83 | 115→98→110 | 强烈支持 GCAD |

三组平均 F1：

| 统计范围 | GCAD-on | GCAD-off | on−off |
|---|---:|---:|---:|
| 上述三组宏平均 | 0.923419 | 0.645722 | +0.277697 |

这个宏平均不能单独作为最终结论，因为：

- SP single→multiple 的巨大差值主导了平均值；
- US night 明确出现负向结果；
- on/off 虽具有相同类别输入和调用次数，但 LLM 每次返回的序列数量仍可能不同；
- 当前每种设置只有一次生成，无法区分 GCAD 效果和 LLM 随机波动。

三组 off 的正式归档均已保存完整模型、SSC、prompt/response、原始与解析生成数据、TOF 中间产物、检测数据和 checksum。

### 6.2 论文级优先项：多次重复

对上述 3 个代表任务分别再执行两次 GCAD-on 和两次 GCAD-off，使每个任务、每种模式达到 3 次独立重复。需新增 **12 个完整运行**，并报告：

- F1、Precision、Recall、Accuracy 的均值 ± 标准差；
- 若检测阶段可导出连续异常分数，再补 AUROC 和 AUPRC；
- 每一重复中 `on - off` 的配对差值，以及平均差值；
- 可选配对统计检验或 bootstrap 置信区间。

当前生成后端不保证完全可由 seed 决定，因此应称为“独立重复运行”，不能暗示 LLM 输出严格确定性。

尤其应观察：

- US winter→spring 的小幅正增益是否稳定；
- US daytime→night 的负增益是否重复出现；
- SP single→multiple 的 off=0 是否为稳定现象，而非单次生成导致。

### 6.3 强证据：随机关系负对照

至少在 US winter→spring 上加入 `shuffled-GCAD`：

- 保持与真实 GCAD 完全相同的关系条数、权重格式和提示词长度；
- 打乱源节点、目标节点或 lag；
- 比较 `real GCAD > shuffled GCAD ≈ GCAD-off` 是否成立。

这个对照能排除“只是提示词多写了几行，所以结果变好”的解释，比单独 on/off 更有说服力。

### 6.4 机制证据

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
- 首轮 on/off 已证明不能把 GCAD 描述为“所有任务必然提升”：US daytime→night 的 off 结果高于 on。
- `experiment-seed=2024` 控制 SSC、TOF、GCAD holdout 和异常检测器，但 Codex CLI 不应用 temperature、top_p、generation seed 和 max_tokens；因此相同调用次数也会产生不同数量和内容的序列。
- FR daytime→night、SP daytime→night 因门控未启用 GCAD，只能说明系统在无可靠关系时可安全退化，不能作为“GCAD 带来提升”的证据。
- FR winter→spring 缺少可选的 TOF candidate scratch 中间文件，但正式复现所需的核心产物、指标与清单完整。

## 8. 推荐执行顺序

1. 首轮 3 个严格配对的 GCAD-off 已完成。
2. 下一步优先新增 12 个 on/off 独立重复，形成每种设置 3 次的均值 ± 标准差。
3. 若重复后 US night 仍为负增益，应分析 11 条注入关系是否与夜间迁移约束冲突，并考虑更严格的关系门控或更小提示剂量；不能删除反例。
4. 在 US winter→spring 增加 shuffled-GCAD 负对照和关系遵守率。
5. 最后再考虑 history、关系上限或稀疏阈值等参数敏感性。

因此，主实验无需推倒重跑；当前最重要的是把首轮消融扩展为多次独立重复，确认两组正效应和一组负效应是否稳定。
