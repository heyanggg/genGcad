# GCAD v2 representation design

R1 constructs coarse continuous event-time tensors within an original source sequence and natural day. A row is one native-derived time slot, simultaneous actions are multi-hot, a real empty slot is retained as an all-zero observed row, and padding is a separate mask concept. Because FR winter has only three-hour bins, tested resolutions are 3, 6, and 12 hours; they are not described as exact continuous timestamps.

R2 maps each legal event in an original independent source sequence to an event position. It preserves order, creates no physical-time claim, and never joins sources. The preregistered formal trial is R2/history 2 because the audit established timestamp insufficiency and it yields 579 windows, above the frozen 330-window minimum. R2 histories 3/4/6 yield 360/243/103 windows. R1 3-hour histories 2/3/4/6 yield 288/216/151/45; 6-hour yields 102/47/0/0; 12-hour yields zero.

All trials and failures remain in `representation_trials.jsonl`. Source provenance is split before window construction.
