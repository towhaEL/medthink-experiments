# 15 — Research Proposal: Rationale-Supervised Bangla Medical VQA (updated 2026-11-24)

This is the single, current statement of the project: what we are doing, why, how the pipeline is improved, which experiments are running, and what is left before submission. It supersedes the framing in docs 04 and 07 where they differ. Evidence: docs 13 and 14. Live run state: doc 09.

---

## 0. Summary in plain terms

- **Problem.** Small, explainable medical VQA models are what low-resource health systems can actually run. MedThink (NAACL Findings 2025) reported that training a small model to also *explain* its answer (a "rationale") makes it more accurate in English. Nobody has tested this in a low-resource language.
- **What we built.** The first Bangla Med-VQA benchmark with human-checked rationales: R-RAD-Bn, R-SLAKE-Bn and R-Path-Bn, about 39k QA pairs, each fully parallel to English.
- **What we found.**
  - Rationale training as MedThink does it **hurts** accuracy: −5 to −14 points in Bangla.
  - Under a careful multi-seed protocol it **also hurts in English** (−4 to −9 points), so MedThink's English gain does not reproduce.
  - The cause is **answer-loss dilution**. The 1–2 token answer is ~2% of the training target, so the loss is almost all rationale.
- **Our fix.**
  - **Balance the loss** so the answer and the rationale each get equal weight.
  - Use a **medical image encoder** (BiomedCLIP) instead of generic COCO features.
  - Result: Bangla RAD 69.0 → **75.2**, SLAKE 78.8 → **87.6**, with rationale quality unchanged. The model runs at 0.25B parameters on a free Kaggle T4.
- **Target.** A Q1 journal (Computers in Biology and Medicine; alternative: Artificial Intelligence in Medicine), submission **~Jan 12–15, 2027**.

---

## 1. Background and problem

| | |
|---|---|
| **Task** | Medical visual question answering (Med-VQA): given a radiology or pathology image and a question, answer it. Closed-end = yes/no or a choice; open-end = free text. |
| **Rationale supervision** | Train the model to output `answer + explanation` (MedThink "Explanation" strategy) using physician-checked rationales. In MedThink: UnifiedQA-T5-base + DETR features + a gated cross-attention fusion, +4.5 / +3.8 / +1.0 accuracy on RAD / SLAKE / Path (English, single seed). |
| **Why Bangla** | ~270M speakers, with few clinical AI resources. Existing Bangla Med-VQA work benchmarks large LVLMs on answer accuracy only. There are no rationales and no small, deployable, explainable models. |
| **Gap** | (1) No rationale-annotated Bangla Med-VQA data. (2) No evidence that rationale supervision transfers to a low-resource language. (3) No analysis of *why* rationale supervision helps or hurts small models. |

---

## 2. Research goal and questions

**Goal:** a small (0.25B), explainable Bangla Med-VQA model that is *more* accurate because of its rationales, not less. We also want to understand the mechanism well enough to give practical guidance for other low-resource languages.

| RQ | Question | How we answer it | Status |
|---|---|---|---|
| **RQ1** | Does rationale supervision help small Med-VQA models in Bangla, as reported in English? | Same pipeline, 3 seeds, w/o-R vs Explanation on 3 datasets (E1) | ✅ **No:** −5.4 (RAD), −14.0 (SLAKE), −9.8 (Path) |
| **RQ2** | Is the effect specific to Bangla, or does it also appear in English? | Same-pipeline English control with identical splits and images (E5) | 🟡 Hurts in English too (RAD −4.3, SLAKE −8.5; 3 seeds); larger in Bangla on SLAKE. Path English running |
| **RQ3** | Why does it hurt? | Loss-share analysis, segment-balanced loss (E2), image ablations (E0), encoder swap (E3), rationale dropout and length (E2/E4) | 🟡 Answer-loss dilution confirmed as the main cause; weak visual grounding is a second one |
| **RQ4** | Can a lightweight fix make rationales help, without worse explanations? | AARL-full (E3c), rationale metrics, clinician ratings (human eval) | 🟡 Accuracy ✅; automatic rationale metrics ✅; clinician ratings not started |
| **RQ5** | How does a 0.25B fine-tuned model compare with zero-shot 3–4B VLMs in Bangla? | Qwen2.5-VL-3B / MedGemma-4B zero-shot (E6) | ⬜ Not built |

---

## 3. Hypotheses and current evidence

| # | Hypothesis | Evidence so far | Verdict |
|---|---|---|---|
| H1 | Rationale supervision hurts in Bangla | 3 seeds, 3 datasets; McNemar p = 0.006 (RAD), 4e-17 (SLAKE), 1e-108 (Path) | ✅ Confirmed |
| H2 | …but helps in English (as MedThink reports) | EN Explanation − w/o-R: RAD −4.3 (p = 0.015), SLAKE −8.5 (p = 3e-9) | ❌ **Rejected:** it also hurts in English under our protocol |
| H2′ | The harm is larger in Bangla | SLAKE −14.0 BN vs −8.5 EN; RAD −5.4 vs −4.3 | 🟡 Partly (SLAKE yes, RAD similar). Path EN decides |
| H3 | The cause is answer-loss dilution | Per-segment normalisation (α=0.5) recovers +11 to +14 pts over vanilla in both languages; vanilla + BiomedCLIP stays low (SLAKE 68.8, RAD 64.0) | ✅ Strongly supported |
| H4 | The visual pathway is under-used with generic features | DETR: shuffled image ≈ real image; Path ignores the image completely. BiomedCLIP on SLAKE: zero image −27 pts | ✅ Supported (SLAKE); RAD/Path limited (see §6) |
| H5 | Opening the gate (bias −1 + warm-up) helps | Training closes the gate again; no gain with BiomedCLIP (85.6 vs 88.5) | ❌ Dropped from the method |
| H6 | Balanced loss keeps rationale quality | ROUGE-L 19.7 vs 20.6 (RAD), 23.2 vs 23.0 (SLAKE); only α ≥ 0.8 degrades | ✅ Automatic metrics; clinician check pending |

---

## 4. Contributions (revised)

| # | Contribution | Type | Status |
|---|---|---|---|
| **C1** | **BnMedThink:** the first human-verified, rationale-annotated, EN–BN parallel Med-VQA benchmark (R-RAD-Bn, R-SLAKE-Bn, R-Path-Bn; ~39k QA + rationales; fixed splits) | Resource | Data ✅; translation-quality statistics + 300-item audit ⬜ |
| **C2** | **Controlled cross-lingual study:** under a multi-seed, early-stopped protocol, standard rationale supervision hurts small Med-VQA models in *both* languages (MedThink's English gain does not reproduce), with a larger drop in Bangla on SLAKE | Empirical | 🟡 RAD/SLAKE done; Path EN running |
| **C3** | **Mechanism:** answer-loss dilution (the answer is ~2% of target tokens, more so with longer rationales), plus under-used visual input; shown by loss balancing, image ablations and encoder swaps | Analysis | 🟡 Mostly done; rationale dropout running, length ablation optional |
| **C4** | **AARL (Answer-Anchored Rationale Learning):** segment-balanced loss (α = 0.5) + BiomedCLIP features. Large, significant gains, and rationales become a net positive on RAD | Method | ✅ RAD/SLAKE 3 seeds; Path running |
| **C5** | **Evaluation suite:** choice scoring, image-blind/shuffled ablations, McNemar over seeds, Bengali-aware rationale metrics, clinician ratings, zero-shot VLM baselines | Evaluation | 🟡 Automatic parts ✅; human eval + VLMs ⬜ |

**Positioning of C4.** Weighting answer and rationale losses separately is related to *Distilling step-by-step* (Hsieh et al., 2023), which we cite. The novelty claim is the **diagnosis** and the **controlled cross-lingual evidence**, not the loss formula. A second finding: a medical encoder pays off only once the loss is fixed (vanilla + BiomedCLIP 68.8 vs AARL-full 87.6 on SLAKE).

**Working title:** *When Rationales Hurt: Answer-Loss Dilution in Rationale-Supervised Medical VQA and a Lightweight Fix for Low-Resource Bangla*

---

## 5. The pipeline: baseline vs improved

```
                 ┌──────────────── MedThink-BN (baseline) ────────────────┐
 image ──► DETR (COCO, 100×256) ──► gated cross-attention (λ≈0.05) ─┐
 question + options (Bangla) ──► BanglaT5 encoder ──────────────────┴─► BanglaT5 decoder
                                                     target: "উত্তর: A  কারণ: <~110-token rationale>"
                                                     loss: token-mean NLL  → answer ≈ 2% of gradient

                 ┌──────────────── AARL-full (ours) ──────────────────────┐
 image ──► BiomedCLIP ViT-B/16 (197×768) ──► gated cross-attention ─┐
 question + options (Bangla) ──► BanglaT5 encoder ──────────────────┴─► BanglaT5 decoder
                                                     same target
                                                     loss: 0.5·L_answer + 0.5·L_rationale
                                                           (each length-normalised per example)
 inference: choice scoring over the options + generated Bangla rationale
```

| Stage | Baseline (MedThink, ported) | Improvement | Why | Effect |
|---|---|---|---|---|
| Data | English R-RAD/SLAKE/Path | Human-verified Bangla translation of Q, options, answer and rationale; EN–BN parallel | Resource + controlled comparison | C1 |
| Text backbone | UnifiedQA-T5-base (English) | BanglaT5 (with a guard against the transformers-5 lm_head re-tie bug) | The English tokenizer cannot represent Bangla | Required |
| Visual features | DETR, COCO-trained | **BiomedCLIP** patch tokens | Medical-domain features that generalise to unseen images | SLAKE +7.9 (w/o-R), +9.2 (with α=0.5) |
| Fusion | Gated cross-attention, gate init λ≈0.05 | Kept (gate warm-up tested and dropped) | Warm-up did not help | — |
| Loss | Token-mean NLL over answer + rationale | **Segment-balanced, α = 0.5** | Removes answer-loss dilution | +11 to +14 over vanilla |
| Training protocol | Fixed epochs, single seed, no val split | Val split, early stopping (patience 15/15/5, min 50% of cap), best-weight restore, **3 seeds** | Fair, reproducible comparison | All numbers |
| Evaluation | Generated-string match, n-gram rationale metrics | Choice scoring, image-blind/shuffled ablations, paired McNemar, Bengali-aware metrics, clinician ratings | Robust and interpretable | C5 |
| Compute | — | Single free Kaggle T4, 0.25B params, 5 co-author accounts in parallel | Deployability argument | — |

---

## 6. Main results so far (closed-end test accuracy, mean ± std over 3 seeds)

| Condition | RAD-Bn | SLAKE-Bn | Path-Bn |
|---|---|---|---|
| Majority class | 51.1 | 56.7 | 54.7 |
| w/o-R (DETR) | 69.0 ± 1.3 | 78.8 ± 1.4 | 83.0 ± 0.5 |
| Explanation, vanilla MedThink (DETR) | 63.6 ± 2.2 | 64.8 ± 1.8 | 73.2 ± 0.8 |
| Explanation + α=0.5 (DETR) | 74.9 ± 0.8 | 78.4 ± 1.9 | 82.6 ± 0.5 |
| w/o-R + BiomedCLIP | 70.1 ± 1.7 | 86.7 ± 1.8 | 81.4 (1 seed) |
| Explanation vanilla + BiomedCLIP | 64.0 (1) | 68.8 (1) | — |
| **AARL-full (α=0.5 + BiomedCLIP)** | **75.2 ± 0.4** | **87.6 ± 1.0** | 83.6 (1 seed) |

**Where the gain comes from** (pooled McNemar):

| | RAD | SLAKE |
|---|---|---|
| BiomedCLIP alone | +1.1 (n.s.) | **+7.9** (p = 3e-11) |
| Balanced rationale loss on top | **+5.1** (p = 0.002) | +0.9 (n.s.) |
| Total vs w/o-R | **+6.2** (p = 1e-4) | **+8.8** (p = 1e-13) |

- **RAD:** 97% of test images also appear in training, so better features add little; the rationale signal is what helps.
- **SLAKE:** test images are unseen, so generalisable medical features are what matter. Rationales come at no cost.
- **Path:** closed questions are largely answerable from text priors, so the image barely matters with either encoder. This is a limitation to state.

**English vs Bangla, same pipeline** (Δ vs w/o-R in the same language, 3 seeds):

| | RAD BN | RAD EN | SLAKE BN | SLAKE EN |
|---|---|---|---|---|
| w/o-R level | 69.0 | 75.9 | 78.8 | 81.4 |
| Vanilla Explanation | −5.4 | −4.3 | **−14.0** | −8.5 |
| Explanation + α=0.5 | **+5.9** | **+4.0** | −0.3 | −0.4 (2 seeds) |

---

## 7. Experiments: what, why and progress (2026-11-24 16:00)

Legend: ✅ done · 🟡 running or partial · ⏳ queued · 🔧 code needed · ⬜ not started

| ID | Experiment | Answers | Runs | Status | Key result / note |
|---|---|---|---|---|---|
| E0 | Image ablation (real / zero / shuffled), built into every run | H4 | all | ✅ | DETR: images barely used; BMC on SLAKE: zero image −27 pts |
| E1 | Baselines w/o-R, Explanation, Reasoning × 3 seeds (RAD, SLAKE, Path) | RQ1 | 23 | ✅ | Explanation −5 / −14 / −10 |
| E2 | Segment loss: α sweep {0.3, 0.5, 0.8, 0.95} + α=0.5 × 3 seeds | H3 | 15 | ✅ | α=0.5 best; gate passed on Nov 22 |
| E2-rd | **Rationale dropout 50%** (+ BMC), the simple alternative to the balanced loss | H3 / reviewer baseline | 1 sanity + 6 | 🟡 sanity queued (code done today) | Run the batch after the sanity check |
| E3a | BiomedCLIP feature extraction | — | 1 | ✅ | 5131 images, (197, 768) |
| E3b | Encoder × gate warm-up 2×2 | H4, H5 | 6 | ✅ | Warm-up dropped |
| E3c | AARL-full × 3 seeds; w/o-R + BMC × 3; vanilla + BMC | C4 decomposition | 12 | ✅ | See §6 |
| E3d | Vanilla Explanation + BMC, seed 43 | H3 | 2 | 🟡 running | s42: RAD 64.0, SLAKE 68.8 |
| E3-Path | Path AARL-full and w/o-R + BMC × 3 seeds | C4 on Path | 6 | 🟡 2 done, 4 running | s42: 83.6 / 81.4 |
| E4 | Rationale length: full / 2 sentences / 1 sentence | C3 (capacity) | 4 | 🔧 optional | Lower priority now that dilution is confirmed |
| E5 | English control, RAD + SLAKE × 3 conditions × 3 seeds | RQ2 | 18 | 🟡 17/18 | See §6; SLAKE EN α=0.5 s43 running |
| E5-Path | English control on Path, s42 | RQ2 (3rd dataset) | 3 | 🟡 running | EN/BN files verified identical in qid, image and answer |
| E6 | Zero-shot VLMs: Qwen2.5-VL-3B, MedGemma-4B; Bangla and English prompts | RQ5 | ~12 evals | 🔧 notebook needed | MedGemma needs HF access |
| E7 | Analyses: consistency (rationale ↔ answer), per organ / question type, error analysis | C3, C5 | inference | 🟡 choice scoring + McNemar done | Breakdowns and consistency to do |
| E8 | Open-end: 3 datasets × {w/o-R, AARL-full} | C4 generality | 6–12 | 🔧 open-end launcher needed | MedThink reports open-end; reviewers will expect it |
| HE-A | Translation quality audit (300 items, 2 clinicians) | C1 | human | ⬜ | ~5 h per annotator |
| HE-B | Clinician rating of rationales: gold vs vanilla vs AARL-full (120 items × 3) | RQ4, H6 | human | ⬜ | ~9 h per annotator; **sheets must go out by ~Dec 5** |
| T2 | Tier 2 (mT5 EN vs BN, EN rationale + BN answer, QLoRA VLM) | Extensions | — | ⬜ | Only if time remains |

**Compute.** Five accounts give 150 GPU-h/week and 10 parallel runs. About 120 runs have finished. Remaining GPU work is ~40 GPU-h, so compute is no longer the bottleneck. Code (E6, E8) and human evaluation are.

---

## 8. Plan to submission

| Week | Dates | GPU / code | People (co-authors, clinicians) |
|---|---|---|---|
| **W1** | Nov 24–30 | Finish Path + BMC, English Path, E3d; rationale-dropout sanity → batch; **build E6** (zero-shot VLMs); **build E8 open-end launcher** | Read the two competitor Bangla Med-VQA papers; request MedGemma access; decide the C2 framing (§10) |
| **W2** | Dec 1–7 | E8 open-end runs (RAD, SLAKE, Path); E6 runs; E7 breakdowns + consistency | **Build human-eval sheets** (HE-A, HE-B) and send to clinicians by Dec 5; draft Dataset + Related Work |
| **W3** | Dec 8–14 | Re-runs/fixes only; optional E4 length ablation; final tables and figures | Clinicians annotate; draft Method |
| **W4** | Dec 15–21 | Tier 2 only if useful | Collect ratings, compute agreement (κ / Krippendorff α); draft Results |
| **W5** | Dec 22–28 | — | Discussion, limitations, abstract |
| **W6** | Dec 29–Jan 4 | — | Internal review by a co-author |
| **W7** | Jan 5–11 | — | Release code + data (GitHub, HF Datasets / Zenodo DOI), cover letter |
| **Submit** | ~Jan 12–15 | | Computers in Biology and Medicine (alt.: AI in Medicine) |

**Paper figures and tables:**
- Tab. 1: dataset + translation-quality statistics.
- Tab. 2: EN vs BN Δ table.
- Tab. 3: main results incl. zero-shot VLMs.
- Tab. 4: ablation/decomposition (loss, encoder, warm-up, dropout).
- Tab. 5: open-end.
- Tab. 6: human evaluation.
- Fig. 2: α trade-off curve (accuracy vs rationale quality).
- Fig. 3: image-ablation drops (DETR vs BMC).
- Fig. 5: per-organ errors.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reviewers see the balanced loss as known (Hsieh et al. 2023) | High | Claim the diagnosis + controlled cross-lingual evidence as the novelty; include the rationale-dropout baseline; cite directly |
| "Bangla is worse" does not hold on all datasets | Medium (RAD similar) | Headline = harm in both languages + dilution mechanism; present Bangla amplification as dataset-dependent |
| Human evaluation slips past the holidays | Medium | Sheets out by Dec 5; 120 items is small; automatic metrics as backup |
| RAD test/train image overlap (97%) | Known | State it as a limitation; use SLAKE as the main visual-generalisation test |
| PathVQA closed answerable from text priors | Known | Report the image ablation; frame Path as a language-prior-dominated dataset |
| Competitor paper (ACL Findings 2026) overlaps | Low–medium | Differentiate: rationales, small model, controlled study; evaluate on their data if released (T2-4) |
| Scheduler stops when the host sleeps | Happened Nov 22–24 | On resume, check `kill -0 $(cat experiments/e3_scheduler.pid)` and restart (safe) |
| Kaggle OAuth for 2 accounts stops refreshing | Happened twice | Re-login; debug KaggleFleet refresh if it recurs |

---

## 10. Decisions needed from the team

1. **C2 framing.** Adopt "rationale supervision hurts in both languages, more in Bangla on SLAKE; the cause is dilution" instead of "helps in English, hurts in Bangla". Recommended: yes, because the English control does not support the old framing.
2. **Open-end (E8).** Include it in the main paper (recommended, ~1–2 days of code) or move it to future work.
3. **Zero-shot VLM baselines (E6).** Qwen2.5-VL-3B only, or also MedGemma-4B (needs HF access).
4. **Clinician time.** Confirm 2 clinicians × ~14 h (HE-A + HE-B) between Dec 5 and Dec 20.
5. **Rationale-length ablation (E4).** Keep it as optional; it adds a figure but no longer changes the conclusion.
