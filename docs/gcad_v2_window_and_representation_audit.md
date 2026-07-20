# GCAD v2 window and representation audit

The historical 165-window result is reproducible and is primarily a representation artifact. FR winter `trn.pkl` contains 873 original independent sequences and 2,917 events. SmartGen TSS preserves the same events but expands them to 1,728 fragments. At the native three-hour resolution, 1,320 fragments span one slot and only 83 fragments span at least five slots, producing exactly 165 history-4 windows.

V1 then randomly divided TSS fragments instead of original sources. The train/validation source overlaps for seeds 2024/2025/2026 were 193, 207, and 198 original sequences. V2 splits the 873 original source IDs before window construction and has zero overlap for all three seeds.

The only timestamp is an integer day and three-hour bin. No continuous timestamp, session timestamp, minute, or second field exists. Continuous-time trials are therefore explicitly coarse-bin ablations. They never cross an original sequence or day. Event-position prediction is the formal representation.

The historical 40-channel vocabulary included `None:location`. Of 2,917 source events, 1,390 carry that metadata action. V2 records and excludes them, leaving 1,527 legal events and 39 legal device-action channels. No illegal node enters the formal vocabulary.

Machine-readable evidence is under `outputs/gcad_source_v2/fr/spring/audit/` and `channels/`. No target behavior was read.
