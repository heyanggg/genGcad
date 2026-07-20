# Known limitations

- CUDA is not usable on this server because PyTorch 2.13.0+cu130 rejects the installed driver (reported version 12060). All real training/TOF/downstream results are CPU results; the conditional CUDA test was skipped honestly.
- The official checkout lacks `SmartGen/IoT_model/Transformer_fr_winter_15epoch.pth`, so SPPC could not be regenerated from scratch. Existing official SPPC day artifacts were used only for prompt representatives. GCAD correctly used the complete TSS output.
- The combined prompt uses all 305 existing representatives instead of reproducing seven independent API prompts. This keeps inputs auditable but is not byte-equivalent to the historical API experiment. Disabled adapter equivalence is tested against the local baseline builder.
- Only FR winter→spring and one Codex generation replicate per prompt arm were run. E/F interfaces exist but E/F generation, other cells, and confidence intervals remain unexecuted.
- Source data are sparse: 1,728 sequences yield only 165 history-4 windows across 40 channels. Many stable edges point to `None:location`, suggesting representation artifacts/degenerate signal.
- The Mixer failed to beat Markov/n-gram validation loss in all five reported replicates. It must not be presented as a predictive improvement.
- Although 39 stable edges survived, only two overlapped existing legal GSS edges; rank changes were zero. Thus the prompt changed largely by appending relation context, not by meaningful original-GSS reordering.
- Downstream enhancement was negative: A F1 0.7733 versus 0.6667 for B/C/D. The latter three had precision 0.5 and recall 1.0, consistent with an overly low generated-only threshold/all-anomaly tendency. No target-informed retuning was performed.
- Generated sequences were authored by the current Codex agent and validated structurally. They are not stochastically reproducible, and no unavailable model/version/token metadata is claimed.
- Role guards provide code-level enforcement, not filesystem sandbox isolation. Target files existed locally and were intentionally opened only in the final evaluation stage.
