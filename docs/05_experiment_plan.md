# 05 — Experiment Plan (Kaggle T4/P100 only, ≈ 4 weeks)

## Compute budget

- Kaggle: 30 GPU-h/week per account → **~120 GPU-h** over 4 weeks for one account. Each co-author's own Kaggle account adds another 30 h/week, which is legitimate and doubles throughput. Assign seeds to co-authors.
- Per-epoch cost, measured from your runs (`metrics_json/*`, elapsed_h / epochs):

| Subset | w/o-R (min/epoch) | Expl/Reasoning (min/epoch) |
|---|---|---|
| RAD closed | 0.6 | 1.9 |
| SLAKE closed | 0.7 | 1.8 |
| Path closed | 3.4 | 11.5 |
| RAD open | 0.5 | 1.4 |
| SLAKE open | 1.0 | 3.0 |
| Path open | 3.1 | 10.3 |

- New epoch caps (doc 03): RAD 40, SLAKE 60, Path 12 with early stopping. Rough cost per closed-end run: RAD 0.4 h (w/o-R) / 1.3 h (rationale), SLAKE 0.7 / 1.8 h, Path 0.7 / 2.3 h, plus ~0.2–0.5 h generation.

All runs: BanglaT5 unless stated; effective batch 32; lr 1e-4; greedy decoding; seeds {42, 43, 44}.

---

## What to rerun vs keep (decision, 2026-11-20)

Do **not** rerun the full 24-run grid. Baselines must be rerun under the new protocol (val split, early stopping, 3 seeds), otherwise gains are confounded with protocol changes. The old grid stays as the appendix "paper-protocol replication".

| Method | Rerun? | Seeds | Subsets |
|---|---|---|---|
| w/o-R | Yes | 3 | closed + open |
| Explanation (vanilla) | Yes | 3 | closed + open |
| Explanation + AARL (main) | Yes | 3 | closed + open |
| Reasoning, vanilla vs + AARL | Light | 1 | closed, RAD + SLAKE |
| Two-Stage | No (appendix only) | – | – |
| Ablations (α, encoder, warm-up, rationale length) | Yes | 1 (best: 3) | closed, RAD + SLAKE |

Full rerun would cost ≈ 200+ GPU-h; this plan is ≈ 90 GPU-h.

## Tier 1 — required for the paper (≈ 85 GPU-h)

### E0. Image-blind and shuffled-image ablation (inference only) — ~1 h
- Reuse saved checkpoints if they are in your Kaggle outputs; otherwise use the new E1 checkpoints.
- Variants: real features / zero features / features of a random other test image.
- Report accuracy and the drop. Expectation, given λ ≈ 0.05–0.15: a small drop for DETR models, a larger drop for AARL. **That contrast is a headline figure.**

### E1. Re-baseline under the fixed protocol — ~27 h
- 3 datasets × closed-end × {w/o-R, Explanation} × 3 seeds, plus Reasoning × 1 seed.
- Purpose: clean main-table baselines with mean ± std, and confirmation that the reversal survives the protocol fix.

### E2. Segment-balanced loss — ~17 h
- RAD + SLAKE closed, Explanation, α ∈ {0.5, 0.8, 0.95}, 1 seed (≈ 9 h).
- Best α: 3 seeds on RAD/SLAKE plus Path × 1 seed (≈ 8 h).
- Also: rationale dropout 50% (one config per dataset, ≈ 3 h, fits within the total if Reasoning is dropped).
- Plot: accuracy vs α, and rationale BLEU/chrF vs α (the trade-off curve).

### E3. Medical visual encoder + gate warm-up — ~15 h
- E3a: extract BiomedCLIP (197 × 768) features for all images (~1 h, one notebook, mirrors notebook 01).
- E3b: on RAD + SLAKE closed with the best α, a 2×2 of {DETR, BiomedCLIP} × {no warm-up, warm-up + bias −1} (≈ 6 h).
  - *Status 2026-11-22:* α=0.5 chosen (doc 13). The DETR / no-warm-up cell reuses the α=0.5 s42 runs, so only 6 new runs are needed. Plan `e3b` has been running since 13:12.
- E3c: best combination = **AARL-full**: 3 seeds on RAD/SLAKE, 1 seed on Path; also run w/o-R + BiomedCLIP (to show whether the gain is vision alone or the rationale) (≈ 8 h).

### E4. Rationale length ablation — ~4 h
- RAD + SLAKE closed, Explanation with α = token-proportional (vanilla): full rationale vs first 2 sentences vs first sentence.
- If accuracy rises as rationales shorten, that supports the capacity explanation (C3).

### E5. Same-pipeline English control — ~5 h (critical)
- `model_name="allenai/unifiedqa-t5-base"`, the English `Medthink-Dataset` JSONs, same DETR features, same protocol, RAD + SLAKE closed, {w/o-R, Explanation} × 1 seed (+2 seeds if time).
- Needs: English markers (`Answer:` / `Because:`) and the English unifier path. Most of `unify_records` already handles this schema.
  - *Status 2026-11-22: built* (doc 08). `overlay_language` swaps the parallel English text into the Bengali manifests, so splits, val items and image features are identical.
  - Plan `e5`: {w/o-R, Expl, Expl + α=0.5} × {RAD, SLAKE}, seed 42, lr 1e-4 (same as BN).
  - α=0.5 was added: it shows whether the loss fix also helps in English. The lr 5e-4 variant is available via `run(..., lr=5e-4)`.
- The result goes into the "Δ(Explanation − w/o-R) EN vs BN" table, the central evidence for C2.
- Also try lr 5e-4 for UnifiedQA (the paper's setting, T5 v1.0 is stable at it).

### E6. Zero-shot VLM baselines (inference only) — ~4 h
- Qwen2.5-VL-3B-Instruct (fp16, ~8 GB) and MedGemma-4B-it (gated on HF: request access now).
- Closed-end RAD/SLAKE full test, Path 1000-item stratified sample. Prompts in Bangla and English (parallel), "answer হ্যাঁ or না".
- Gives the "0.25B fine-tuned vs 3–4B zero-shot" comparison and an EN–BN gap measurement for LVLMs.

### E7. Inference-only analyses — ~3 h
- **Choice scoring** accuracy for every closed-end model.
- **Rationale–answer consistency:** feed each Explanation model's generated rationale to the stage-2 model (it was trained to read rationales and output answers). Consistency = stage-2 answer equals the model's own answer. This reuses existing models and gives a faithfulness metric.
- **Seen/unseen answer split** for open-end.
- Per-question-type / organ / modality breakdowns (SLAKE has `content_type`, RAD has `question_type`), reproducing the paper's Table 4 with AARL.

### E8. Open-end with the final model — ~10 h
- 3 datasets × {w/o-R, AARL-full} × 1 seed (3 seeds for RAD/SLAKE if time).
- Report ans/EM, ans/token-F1 (seen/unseen), rationale chrF + BanglaBERT-BERTScore.

---

## Tier 2 — only if Tier 1 is on schedule (≈ 30 h)

| ID | Experiment | Cost | Value |
|---|---|---|---|
| T2-1 | mT5-base on EN and BN (RAD + SLAKE closed, w/o-R + Explanation): backbone held constant across languages | ~17 h | Cleanest C2 evidence |
| T2-2 | Cross-lingual rationale: BN question → BN answer + **EN rationale** (mT5) | ~6 h | Opens direction B |
| T2-3 | QLoRA Qwen2-VL-2B on RAD-Bn closed | ~8 h | Reference row "small VLM fine-tuned" |
| T2-4 | Evaluate best model on a competitor Bangla Med-VQA set (if released) | ~1 h | Generalisation |

---

## Tables and figures the experiments produce

| Paper item | Source |
|---|---|
| Tab. 1: Dataset statistics + translation-QA statistics | Data + doc 06 |
| Tab. 2: Δ(rationale − w/o-R), EN vs BN, same pipeline | E1 + E5 (+T2-1) |
| Tab. 3: Main closed-end results (mean ± std): zero-shot VLMs, MedThink-BN strategies, AARL | E1, E3c, E6 |
| Tab. 4: AARL ablation (loss / encoder / warm-up) | E2, E3b |
| Tab. 5: Open-end results (ans EM/F1, seen/unseen; rationale chrF/BERTScore) | E8 |
| Tab. 6: Human evaluation (5 criteria, Krippendorff's α) | doc 06 |
| Fig. 1: Pipeline + data example (BN/EN parallel) | – |
| Fig. 2: Accuracy vs α trade-off curve | E2 |
| Fig. 3: Gate λ histograms DETR vs AARL + blind-ablation drop | E0, E3 |
| Fig. 4: Accuracy vs rationale length | E4 |
| Fig. 5: Per-organ / question-type error rates | E7 |
| Appendix: paper-protocol 24-run replication | Existing results |

## Run order (dependencies)

```
code changes (doc 03) ──► E0 ─┐
                  E3a (features) ─┐
E1 ──► E2 (α) ──► E3b ──► E3c ──► E8
          └► E4         E5, E6, E7 run in parallel on spare quota
```
