# Bengali MedThink — results in the paper's table format

Backbone `csebuetnlp/banglat5` (247M) replacing UnifiedQA/T5-base; DETR-101 visual encoder (100x256); lr 1e-4 (not the paper's 5e-4, which diverges on a T5 v1.1 backbone); effective batch 32; greedy decoding.

## Table 2 — Accuracy (%) on Closed-End Questions

Deltas are against each method's own w/o-R row, following the paper.

| Methods | R-RAD | R-SLAKE | R-Path | Parameters |
|:---|---:|---:|---:|:---|
| MedThink (w/o R) *(English, paper)* | 79.0 | 82.5 | 86.0 | 0.2B |
| MedThink (w/ Reasoning) *(English, paper)* | 73.9 (-5.1) | 80.8 (-1.7) | 83.1 (-2.9) | 0.2B |
| MedThink (w/ Two-Stage Reasoning) *(English, paper)* | 80.5 (+1.5) | 79.1 (-3.4) | 87.2 (+1.2) | 0.2B |
| MedThink (w/ Explanation) *(English, paper)* | 83.5 (+4.5) | 86.3 (+3.8) | 87.0 (+1.0) | 0.2B |
| | | | | |
| **Ours-Bengali (w/o R)** | 75.0 | 82.2 | 85.0 | 0.25B |
| **Ours-Bengali (w/ Reasoning)** | 65.1 (-9.9) | 68.5 (-13.7) | 76.1 (-8.9) | 0.25B |
| **Ours-Bengali (w/ Two-Stage Reasoning)** | 64.3 (-10.7) | 66.8 (-15.4) | 81.9 (-3.2) | 0.25B |
| **Ours-Bengali (w/ Explanation)** | 71.7 (-3.3) | 73.8 (-8.4) | 80.0 (-5.0) | 0.25B |
| Majority-class baseline | 51.1 | 56.7 | 54.7 | -- |

† run did not reach its full epoch budget.


## Table 3 — Score (%) on Open-End Questions (full/*)

Whole generation vs whole target -- the variant the paper's Table 3 reports (its BLEU-4 of ~10 is unreachable from 1-2 token answers alone).

| Dataset | Strategy | Rouge-1 | Rouge-2 | Rouge-L | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **R-RAD** | | | | | | | | |
| | w/o R | 63.8 | 39.8 | 63.6 | 51.2 | 39.8 | 32.2 | 26.0 |
| | Reasoning | 24.7 | 6.4 | 18.3 | 21.3 | 10.8 | 5.8 | 3.3 |
| | *Reasoning (English, paper)* | *49.8* | *20.3* | *29.3* | *37.8* | *22.7* | *14.0* | *8.9* |
| | Two-Stage Reasoning | 44.8 | 9.1 | 44.8 | 27.7 | 13.8 | 0.0 | 0.1 |
| | *Two-Stage Reasoning (English, paper)* | *49.1* | *19.9* | *28.7* | *37.7* | *22.5* | *13.9* | *8.8* |
| | Explanation | 24.7 | 6.3 | 18.2 | 22.3 | 11.3 | 6.3 | 3.7 |
| | *Explanation (English, paper)* | *50.2* | *20.2* | *29.5* | *38.3* | *22.9* | *14.0* | *8.8* |
| **R-SLAKE** | | | | | | | | |
| | w/o R | 77.5 | 60.0 | 77.2 | 73.4 | 61.5 | 49.9 | 39.1 |
| | Reasoning† | 36.9 | 12.6 | 26.0 | 32.6 | 19.1 | 11.6 | 7.4 |
| | *Reasoning (English, paper)* | *53.5* | *22.8* | *32.1* | *39.5* | *24.3* | *15.5* | *10.0* |
| | Two-Stage Reasoning | 68.2 | 46.1 | 68.2 | 55.1 | 40.3 | 24.8 | 7.5 |
| | *Two-Stage Reasoning (English, paper)* | *53.2* | *23.1* | *32.0* | *39.5* | *24.5* | *15.8* | *10.3* |
| | Explanation† | 37.2 | 13.7 | 26.2 | 33.2 | 20.2 | 13.2 | 8.8 |
| | *Explanation (English, paper)* | *53.1* | *22.7* | *31.7* | *39.2* | *24.1* | *15.4* | *9.9* |
| **R-Path** | | | | | | | | |
| | w/o R | 47.8 | 15.7 | 47.8 | 31.7 | 14.8 | 6.9 | 2.9 |
| | Reasoning | 22.3 | 4.3 | 16.7 | 15.6 | 6.8 | 3.4 | 1.8 |
| | *Reasoning (English, paper)* | *41.5* | *13.0* | *24.8* | *31.8* | *17.0* | *9.6* | *5.7* |
| | Two-Stage Reasoning | 47.1 | 11.9 | 47.1 | 30.5 | 14.0 | 8.3 | 0.1 |
| | *Two-Stage Reasoning (English, paper)* | *41.7* | *13.2* | *24.9* | *32.1* | *17.1* | *9.7* | *5.8* |
| | Explanation | 20.5 | 4.2 | 16.1 | 13.3 | 5.9 | 3.1 | 1.7 |
| | *Explanation (English, paper)* | *41.9* | *13.2* | *25.0* | *32.1* | *17.1* | *9.7* | *5.8* |

† run did not reach its full epoch budget.


## Table 3b — Score (%) on Open-End Questions (ans/*)

Parsed answer vs gold answer -- the clinically meaningful variant. Not directly comparable to the paper.

| Dataset | Strategy | Rouge-1 | Rouge-2 | Rouge-L | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **R-RAD** | | | | | | | | |
| | w/o R | 45.5 | 24.1 | 45.3 | 36.5 | 29.6 | 23.8 | 18.9 |
| | Reasoning | 22.8 | 7.4 | 22.8 | 7.0 | 3.9 | 1.9 | 0.0 |
| | Two-Stage Reasoning | 9.5 | 0.0 | 9.5 | 3.7 | 0.0 | 0.0 | 0.1 |
| | Explanation | 26.0 | 11.2 | 26.0 | 16.1 | 10.1 | 6.2 | 0.1 |
| **R-SLAKE** | | | | | | | | |
| | w/o R | 63.9 | 18.2 | 63.6 | 59.5 | 47.9 | 38.5 | 31.9 |
| | Reasoning† | 63.4 | 18.0 | 63.2 | 49.2 | 36.1 | 24.5 | 17.3 |
| | Two-Stage Reasoning | 48.8 | 9.4 | 48.7 | 33.0 | 18.3 | 4.1 | 0.1 |
| | Explanation† | 65.3 | 19.1 | 65.1 | 59.9 | 48.7 | 38.1 | 31.4 |
| **R-Path** | | | | | | | | |
| | w/o R | 16.6 | 2.4 | 16.6 | 8.1 | 3.8 | 1.7 | 0.9 |
| | Reasoning | 8.6 | 1.5 | 8.6 | 4.9 | 2.4 | 0.0 | 0.0 |
| | Two-Stage Reasoning | 12.3 | 1.7 | 12.3 | 4.9 | 3.3 | 0.0 | 0.0 |
| | Explanation | 14.4 | 2.1 | 14.4 | 4.8 | 2.0 | 0.0 | 0.0 |

† run did not reach its full epoch budget.


## Table 4 — Error rate per anatomical region, R-SLAKE closed-end (lower is better)

| Anatomical Regions (Number) | w/o Rationale ↓ | Explanation ↓ | *paper w/o R* | *paper Expl.* |
|:---|---:|---:|---:|---:|
| Lung (N=141) | 13.48% | 21.99% | *12.06%* | *9.93%* |
| Abdomen (N=141) | 21.99% | 26.95% | *24.11%* | *19.15%* |
| Head (N=91) | 17.58% | 28.57% | *18.68%* | *13.18%* |
| Neck (N=16) | 12.50% | 31.25% | *18.75%* | *6.25%* |
| Chest (N=5) | 0.00% | 0.00% | *20.00%* | *0.00%* |
| Pelvic Cavity (N=22) | 27.27% | 40.91% | *4.55%* | *13.64%* |
| **All (N=416)** | **17.79%** | **26.20%** | -- | -- |

Region labels appear in both Bengali and English spellings in R-SLAKE; they are merged to the paper's six regions, which reproduces its exact counts (141/141/91/16/5/22).

