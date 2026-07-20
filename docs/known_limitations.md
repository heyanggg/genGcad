# Known limitations

Support-aware selection can only retain evidence present in the frozen candidate pool. If Codex authorship does not produce the uniform pool targets, selection cannot manufacture missing support and the replicate must fail.

Replicate-5 met the pool and selected support targets, but its generated-validation reconstruction losses remained dominated by the top 10% of samples and its fixed-split thresholds were unstable. It failed before target access, so A5 supplies no target metric. A later, explicitly preregistered pure-source GCAD v2 audit was allowed to study mechanism viability without rewriting that failure.

- GCAD v2 found only coarse day/three-hour timestamps, not continuous event timestamps. Event-position/history 2 yields 579 windows, but histories 4/6 fall to 243/103. Rare actions remain extremely sparse.
- V2 eliminates `None:location` and source leakage, yet the selected Mixer loses macro F1 to n-gram statistics on all three fixed seeds. This blocks formal relations and is the decisive reason to terminate the current GCAD route.

- GPU visibility changed during the project. Replicate-5 preserves one invalid CUDA TOF attempt for audit, then explicitly hid CUDA for the formal TOF and reconstruction runs. Device selection must be forced rather than inferred from earlier availability checks.
- The official checkout lacks `SmartGen/IoT_model/Transformer_fr_winter_15epoch.pth`, so SPPC could not be regenerated from scratch. Existing official SPPC day artifacts were used only for prompt representatives. GCAD correctly used the complete TSS output.
- A1 combined all 305 representatives into one prompt. A2 now restores the 15 official source groups, but remains non-equivalent to the historical generation experiment because the official backend was GPT-4o, no explicit official output count existed, and A2 uses offline Codex-authored files.
- Only FR winter→spring and one Codex generation replicate per prompt arm were run. E/F interfaces exist but E/F generation, other cells, and confidence intervals remain unexecuted.
- Source data are sparse: 1,728 sequences yield only 165 history-4 windows across the v1 representation. V1's `None:location` edges were invalid; semantic-channel validation now rejects such artifacts at tensorization, stable-relation, fusion, and prompt boundaries rather than treating them as signal.
- The v1 Mixer failed to beat Markov/n-gram validation loss in all five reported replicates. V2's capacity/loss grid also lost the preregistered macro-F1 comparison on all three seeds. It must not be presented as a predictive improvement.
- Although 39 stable edges survived, only two overlapped existing legal GSS edges; rank changes were zero. Thus the prompt changed largely by appending relation context, not by meaningful original-GSS reordering.
- Downstream v1 enhancement was negative: A1 F1 0.7733 versus 0.6667 for B/C/D. Ranking-only was methodologically confounded by `WeightedRandomSampler(replacement=True)`; the formal path now uses full-coverage per-sample weighted loss, but C/D were not rerun in this phase.
- A2 removed action-template collapse and passed the source/internal gates, but held-out reconstruction was bimodal and its 95.5th-percentile threshold rose to 4.292. Final precision/recall/F1/FPR were 0.7647/0.4432/0.5612/0.1364. The threshold exceeded the target attack median (3.2847), causing 49 false negatives. No target-informed retuning was performed.
- Generated sequences were authored by the current Codex agent and validated structurally. They are not stochastically reproducible, and no unavailable model/version/token metadata is claimed.
- Role guards provide code-level enforcement, not filesystem sandbox isolation. Target files existed locally and were intentionally opened only in the final evaluation stage.
- A2 has one authored generation replicate. Its perfect action-template uniqueness is too broad relative to observed source behavior; the original pre-target gates did not measure this dimension.
- `source_semantic_v1` now supplies that gate and A2 fails it post hoc. Because the policy was adopted after A2 target metrics were viewed, it cannot retroactively make A2 target-blind. The frozen replicate-2 requests must be treated as a new protocol artifact, and no target evaluation should occur until an independently authorized generation run passes all pre-TOF gates.
# Replicate-4 source-support failure

The formal GPT-5.6 Codex-agent replicate-4 stopped before TOF because two actions had support in only three independent sequences, below the frozen minimum four. Token repetition inside a sequence does not provide independent training support.
