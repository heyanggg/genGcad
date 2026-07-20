# Third-party notices

## SmartGen

Source: `https://github.com/horizonsinzqs/SmartGen`; local baseline commit `c2ed36c`; MIT license, copyright 2025 horizonsinzqs (see `LICENSE`). Original TSS, SPPC/SSC, GSS, prompt/API, two-stage TOF, and downstream detector remain in their original modules. Compatibility changes are limited to package imports, device-neutral `.to(device)`/checkpoint loading, explicit-path wrappers, and optional soft sampling; original entry points remain available.

## GCAD

Source: `https://github.com/Tc99m/GCAD`; audited local source `/home/heyang/projects/GCAD`; MIT license, copyright 2025 Zehao Liu, Mengzhou Gao and Pengfei Jiao. Referenced files/classes/functions: `models/tsmixer.py::TSMixerRevIN`, `models/common.py::ResBlock`, `utils/dataloader.py` window loaders, and `test.py::save_train_mean_causal`/`test` per-output gradients and asymmetric relation logic.

Corresponding new adaptations are `SmartGen/gcad_source/mixer_predictor.py`, `window_dataset.py`, `gradient_relation.py`, `asymmetric_filter.py`, and `stability_filter.py`. They adapt continuous CSV/RevIN/MSE anomaly scoring to independent SmartGen event sequences, multi-hot BCE, lag-preserving relations, source-only replicate stability, conservative GSS fusion, and generation ranking. The whole GCAD repository was not copied, its source checkout was not modified, and this project claims only a **GCAD-style predictive directional relation**, not full GCAD reproduction or physical causality.
