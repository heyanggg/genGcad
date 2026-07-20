# Local GCAD source audit

`GCAD_SOURCE_DIR=/home/heyang/projects/GCAD`. It is the clean `master` checkout of `https://github.com/Tc99m/GCAD.git`, identified from README content and remote, not from its directory name alone. It is the official AAAI-25 GCAD code and uses the MIT license (copyright 2025 Zehao Liu, Mengzhou Gao and Pengfei Jiao).

## What the local code actually does

- `utils/dataloader.py::SwatDataLoader_AD` reads `train.csv` and `test.csv` (lines 27–35), splits source train 80/20 (37–45), fits scaling only on train (47–57), and builds sliding windows through `CustomDataset`/subsets (67–108, class at 297). Its source format is continuous CSV `[time, features..., label]`, unlike independent SmartGen event sequences.
- `models/tsmixer.py::TSMixerRevIN` (8–38) uses RevIN, repeated `ResBlock`s, and a time-axis linear prediction head. `models/common.py::ResBlock` supplies temporal mixing, feature mixing, and residual additions. The SmartGen adapter retains these mixer/residual ideas but replaces RevIN/continuous regression with multi-hot next-bin logits.
- `main.py::main` constructs the loader and `TSMixerRevIN` (23–50), trains with global MSE (52–142), checkpoints the best validation model (131–146), builds the normal relation pattern, and then evaluates anomalies (150–152).
- Crucially, `test.py::save_train_mean_causal` computes one error per output feature (138–145), backpropagates it to the history input, and stacks a tensor shaped `[batch,input_lag,input_feature,output_feature]` (147–166). It averages over lag (172–174), takes pairwise asymmetric differences (176–182), thresholds weak values (186–199), and saves the train relation matrix.
- `test.py::test` repeats per-output gradients for test windows (220–273), performs the same asymmetric construction (278–291), then compares test relation patterns with the saved normal pattern to score anomalies (294 onward). Thus the upstream matrix is input-feature × output-feature after lag aggregation; its test objective is relation-change anomaly detection.

## Adaptation boundary

Directly portable ideas: TSMixer residual temporal/feature mixing, per-output loss gradients, input→output orientation, asymmetric subtraction, and sparse normal-pattern construction. Required rewrites: independent-sequence windowing, categorical device-action tensorization, BCE per-channel loss, masks, lag-specific preservation, multi-seed/source-partition stability, SmartGen GSS fusion, and generation ranking. The integration does not copy the whole project and does not claim a full GCAD reproduction. Corresponding files are `mixer_predictor.py`, `window_dataset.py`, `gradient_relation.py`, `asymmetric_filter.py`, and `stability_filter.py`.

The reference checkout was never edited; final `git -C /home/heyang/projects/GCAD status --short --branch` reports `## master...origin/master` with no changes.
