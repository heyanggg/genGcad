# SmartGen official generation-protocol audit

Audit scope: official baseline commit `c2ed36c`, FR winter→spring, SPPC threshold 0.918. This document distinguishes protocol structure from the contents of the historical GPT-4o synthetic data.

## Real organization

1. `SmartGen/split.py::Split` writes the complete TSS output `SmartGen/IoT_data/fr/winter/split_trn.pkl` (lines 115–129).
2. `SmartGen/dayse.py::Dayse` groups complete sequences by the first day value and writes `trn_day_0.pkl` through `trn_day_6.pkl` (lines 4–22).
3. `SmartGen/sppc.py::SPPC_select` independently selects representatives for each of seven days and writes `trn_day_<day>_SPPC_th=0.918.pkl` (lines 91–135). These precomputed artifacts exist and contain 37, 46, 40, 39, 69, 38 and 36 sequences, 305 total. The referenced training checkpoint `SmartGen/IoT_model/Transformer_fr_winter_15epoch.pth` is absent, so SPPC cannot be retrained exactly, but retraining is unnecessary for A2 because the official intermediate outputs are present.
4. `SmartGen/find_categories.py::Find_categories` does **not** send an entire day larger than 30 representatives in one prompt. It deterministically partitions each day into consecutive groups of at most 30 (lines 4–28). For FR/0.918 this yields 15 groups: `0_0,0_1,1_0,1_1,2_0,2_1,3_0,3_1,4_0,4_1,4_2,5_0,5_1,6_0,6_1`, with representative counts `30,7,30,16,30,10,30,9,30,30,9,30,8,30,6`.
5. The corresponding numeric and text group artifacts already exist under `SmartGen/IoT_data/fr/winter/trn_day_<group>_SPPC_th=0.918{,_text}.pkl`. `SmartGen/transtext.py::Transtext` converts each group independently (lines 4–26).
6. `SmartGen/main.py` loops over `all_categories` and builds one independent prompt per group (lines 117–151). Every prompt contains: target static device/action text, the winter→spring context sentence, only that group's representative sequences, and the common source GSS. `LLM_call` is invoked once per group and the raw response is saved separately (lines 152–154). No explicit output count is stated in the official prompt; the number returned by GPT-4o is variable.
7. `SmartGen/extract.py::Extract` parses every group response independently (lines 6–48). `SmartGen/transnumber.py::Transnum` then concatenates the parsed groups in `all_categories` order (lines 17–55), converts quadruples, removes illegal quadruplets containing 99999, and writes the aggregate pre-TOF PKL.
8. `SmartGen/security_check.py::security_check` applies the original two-stage TOF to the aggregate.

## Audited FR counts

The 15 historical group outputs contain `8,6,12,10,7,6,6,8,8,29,9,9,6,7,6` sequences, totaling 137 before TOF. The official filtered downstream artifact contains 125. These counts establish experiment scale and per-group allocation for the Codex-file protocol; historical sequence contents and their length distribution are not reused as A2 inputs or generation targets.

The historical aggregate has length 2–10 only as an observed outcome of independent GPT-4o group calls and subsequent conversion. The official prompt itself specifies only that sequences must contain more than one behavior; it does not impose 2–10. A2 therefore derives explicit allowed lengths from each source representative group, not from target behavior or historical GPT-4o length frequencies.

## A1 deviation and A2 recovery

A1 loaded the seven day-level representative files, concatenated all 305 sequences, constructed one 55 KB prompt, and divided a single generation job into seven count batches. This removed the official 15 independent prompt contexts and encouraged globally repeated templates. A2 restores the exact 15 source group boundaries and group-local prompts. It remains a **SmartGen-compatible Codex-file baseline**, not an official generation reproduction: the official backend was GPT-4o, while A2 responses are authored through the offline Codex agent file protocol, and the historical prompt did not fix per-group output counts.

All group source paths and hashes are persisted in the A2 request manifest. No target normal or attack file is used in this audit or request construction.
