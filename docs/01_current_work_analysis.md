# 01 — Analysis of the Current Work (Bengali MedThink)

Date of analysis: 2026-11-20
Scope: everything in `/home/tn-99778/Towha/kaggle` (code, data, metrics, predictions, draft tables).

---

## 1. What you have built

| Asset | Status | Notes |
|---|---|---|
| **R-RAD-Bengali, R-SLAKE-Bangla, R-PathVQA-Bengali** | Done | LLM-translated, fully human-checked Bengali versions of MedThink's R-RAD / R-SLAKE / R-Path. Question, choices, answer **and the medical decision-making rationale (MDMR)** are translated. |
| Parallel English originals (`Medthink-Dataset/`) | Present | Same record IDs, same images → a **parallel EN–BN corpus**. This is a big under-used asset (see `04_contributions_and_directions.md`). |
| `medthink_bn.py` | Done | Clean re-implementation: data unification, DETR (100×256) gated cross-attention fusion, BanglaT5 backbone, Bengali-aware BLEU/ROUGE/chrF/TokenF1. |
| 24-run replication grid | Done (2 runs truncated) | 3 datasets × 2 subsets × 4 strategies, 1 seed. |
| Paper-format tables | Done | `paper_tables/paper_format_tables.md`. |

### Dataset size (Bengali)

| Dataset | Closed train / val / test | Open train / val / test |
|---|---|---|
| R-RAD-Bn | 1823 / – / 272 | 1241 / – / 179 |
| R-SLAKE-Bn | 1943 / – / 416 | 2976 / – / 645 |
| R-Path-Bn | 9428 / 3011 / 3257 | 9680 / 3064 / 3292 |
| **Total** | | **≈ 39.2k QA pairs, each with a Bengali rationale** |

---

## 2. Results so far (BanglaT5, single seed)

### Closed-end accuracy (%)

| Strategy | R-RAD | R-SLAKE | R-Path | English paper (RAD/SLAKE/Path) |
|---|---|---|---|---|
| w/o rationale | **75.0** | **82.2** | **85.0** | 79.0 / 82.5 / 86.0 |
| Explanation (A→R) | 71.7 (−3.3) | 73.8 (−8.4) | 80.0 (−5.0) | 83.5 (+4.5) / 86.3 (+3.8) / 87.0 (+1.0) |
| Reasoning (R→A) | 65.1 (−9.9) | 68.5 (−13.7) | 76.1 (−8.9) | 73.9 / 80.8 / 83.1 |
| Two-Stage | 64.3 (−10.7) | 66.8 (−15.4) | 81.9 (−3.2) | 80.5 / 79.1 / 87.2 |
| Majority class | 51.1 | 56.7 | 54.7 | – |

### Open-end (answer-only token F1 = `ans/ROUGE-1`)

| Strategy | R-RAD | R-SLAKE | R-Path |
|---|---|---|---|
| w/o rationale | **45.5** | 63.9 | **16.6** |
| Explanation | 26.0 | 65.3† | 14.4 |
| Reasoning | 22.8 | 63.4† | 8.6 |
| Two-Stage | 9.5 | 48.8 | 12.3 |

† SLAKE open-end Explanation/Reasoning stopped at 207/300 and 209/300 epochs.

---

## 3. Key findings from my analysis (new evidence, not yet in your tables)

### F1. The rationale benefit reverses in Bengali, and the drop is statistically significant
In English, MedThink's Explanation strategy improves accuracy over w/o-R on all three datasets. In Bengali it **reduces accuracy on all three**. Every rationale strategy is below w/o-R in all 6 (dataset × subset) cells, except SLAKE open-end Explanation (+1.4, from a truncated run).

I ran a paired McNemar test on the per-sample predictions you saved for R-SLAKE closed-end (`pred/`):

| | Expl correct | Expl wrong |
|---|---|---|
| **w/o-R correct** | 286 | **56** |
| **w/o-R wrong** | **21** | 53 |

χ² = 15.0 (with continuity correction), p ≈ 1e-4. The drop is real, not seed noise. **This reversal is your central scientific finding.**

### F2. Loss dilution: the answer is ~2% of the training signal
HF's `T5ForConditionalGeneration` loss is the mean over **all** target tokens (`medthink_bn.py:1209`). Answers are 1–2 words and rationales are 64–79 words (median), p95 = 103–123 words. So in Explanation/Reasoning the answer gets roughly 1–3% of the gradient.

The training curves (`metrics_json/*`, final epoch loss) show this:

| Run | final loss, w/o-R | final loss, Explanation |
|---|---|---|
| RAD closed | 0.002 | 2.09 |
| SLAKE closed | 0.000 | 1.01 |
| Path closed | 0.021 | 2.63 |

w/o-R fits the training set almost perfectly. With rationales, the loss plateaus at 1–2.6 nats per token: a 247M Bengali model cannot fit ~150-token Bengali medical rationales. The loss is spent on rationale tokens it cannot model, and the answer is learned as a side effect. The English model (UnifiedQA, pretrained on far more data) does not hit this capacity wall. This is a **mechanistic, testable explanation** of F1.

### F3. The model barely uses the image
The gate λ starts at sigmoid(−3) ≈ 0.047. After training, the mean λ over all runs is **0.05–0.15**, and the shift from init is **0.01–0.10** (`all_results.csv`, `gate/*`). A few tokens open fully (λ_max ≈ 0.6–1.0), but on average more than 85–95% of the fused representation is still pure text.

Two consequences:
1. Much of the accuracy may come from **language priors**. Reviewers will ask for an image-blind ablation, and you do not have one yet.
2. Rationales describe visual findings ("bright signal on DWI…"). If the model cannot see, it cannot ground the rationale, so rationale tokens become hallucination training. This compounds F2.

Confound: the closed-gate init (bias −3) was chosen for stability. It may also be part of why vision stays closed. This has to be ablated.

### F4. The visual encoder is generic (COCO DETR)
DETR-ResNet-101 trained on COCO object detection is inherited from MM-CoT. Its 100 object queries are tuned to natural-image objects, not to CT, MRI or histopathology. A medical encoder (BiomedCLIP, PubMedCLIP, MedSigLIP) is the obvious, cheap upgrade and also a contribution in itself.

### F5. Rationales leak the answer
The gold answer string appears in the rationale for **74–75% of open-end items** and 38–46% of closed-end items (for yes/no, this counts any occurrence of হ্যাঁ/না, so it is noisy). For Two-Stage, stage 2 is mostly an answer extractor over a stage-1 rationale. Stage-1 rationales have ROUGE-L ≈ 21–24, so they are mostly wrong, and stage 2 inherits the errors. This explains the Two-Stage collapse.

### F6. Data-split facts reviewers will check

| | testQ string seen in train | test image seen in train | test answers unseen in train |
|---|---|---|---|
| RAD closed | 13% | **97%** | 0% |
| RAD open | 11% | **94%** | **54%** |
| SLAKE closed | 62% | 0% | 0% |
| SLAKE open | 67% | 0% | 10% |
| Path closed | 30% | 0% | 0% |
| Path open | 75% | 0% | **36%** |

- VQA-RAD's image overlap is a known property of the original split. State it rather than hide it.
- 54% (RAD) and 36% (Path) of open-end test answers never occur in training, which explains the very low Path open-end scores. Report a **seen/unseen answer breakdown**.

### F7. Protocol weaknesses (fixable)
1. **One seed** per configuration and no confidence intervals.
2. **No validation split** for RAD/SLAKE, and fixed 150/300 epochs. w/o-R reaches loss ≈ 0.1–0.2 within the first 10% of epochs, so 90% of compute is spent overfitting. Path has a val split that is not used for model selection.
3. **The comparison to English uses the paper's numbers.** Those come from a different code base, learning rate (5e-4 vs 1e-4), backbone (UnifiedQA v1.0) and Path split. The "language gap" is therefore confounded with implementation. A same-pipeline English control is needed (see experiment E5).
4. **Two-Stage open-end `full/*` is not comparable to the other rows.** Its full/BLEU-4 is 0.1 while ROUGE-1 is 44.8, which suggests stage 2's target is answer-only. Do not put it in the same column as Explanation/Reasoning full-sequence scores.
5. SLAKE open-end Explanation/Reasoning are truncated (207/300, 209/300).
6. BLEU/ROUGE on answer+rationale measures rationale n-gram overlap, not clinical correctness. Your README already notes this. The paper needs a human or model-based evaluation of rationales.

---

## 4. Strengths to keep and emphasise

- The implementation is careful: schema traps, Bengali metric tokenisation, NaN guards, dynamic padding. Several of these (Bengali BLEU/ROUGE handling, the PathVQA open-end answer bug) are worth a short "evaluation pitfalls for Bengali" subsection. They count as an engineering contribution.
- The parameter budget stays comparable (0.25B vs 0.2B).
- **The data is fully human-checked**, which competing Bangla Med-VQA papers mostly cannot claim for rationales.
- Parallel English and Bengali data, together with 2+ available clinicians, allow controlled experiments that no existing Bangla Med-VQA paper has run.

## 5. Bottom line

As it stands, the work is a **replication with a negative result**. That is not enough for a Q1 journal. The reversal (F1) with its diagnosis (F2–F5) is a genuinely interesting finding, though. If you (a) prove it with a controlled cross-lingual experiment, (b) explain it mechanistically, and (c) fix it with a lightweight method, it becomes a strong paper. See `04_contributions_and_directions.md` and `05_experiment_plan.md`.
