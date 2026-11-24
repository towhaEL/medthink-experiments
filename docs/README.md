# Bangla Med-VQA (MedThink-BN): Research Plan — Index

Prepared 2026-11-20. Target: Q1 journal submission mid-January 2027.
Constraints given: Kaggle T4/P100 only; dataset LLM-translated and fully human-checked; 2+ Bengali-speaking clinicians available.

| Doc | Contents |
|---|---|
| [01_current_work_analysis.md](01_current_work_analysis.md) | What exists, current results, **7 new diagnostic findings** (with numbers) |
| [02_related_work_and_positioning.md](02_related_work_and_positioning.md) | MedThink summary, competing Bangla Med-VQA papers, the gap you fill |
| [03_issues_and_required_fixes.md](03_issues_and_required_fixes.md) | Reviewer-facing weaknesses, protocol changes, code changes (with loss code) |
| [04_contributions_and_directions.md](04_contributions_and_directions.md) | Paper story, contributions C1–C5, the AARL method, ranked research directions, fallback plan |
| [05_experiment_plan.md](05_experiment_plan.md) | Experiments E0–E8 + Tier 2, GPU-hour budget, tables/figures map |
| [06_human_evaluation_protocol.md](06_human_evaluation_protocol.md) | Translation audit + clinician rationale evaluation design |
| [07_timeline_and_paper_outline.md](07_timeline_and_paper_outline.md) | Week-by-week plan, decision gate, target journals, paper outline |
| [13_v2b_decision_gate.md](13_v2b_decision_gate.md) | **Decision gate passed (3 seeds):** α=0.5 segment loss: RAD +5.9 over w/o-R (p=6e-5), SLAKE tie (−0.3, p=0.82); vanilla Explanation −5 / −14 |
| [14_e3c_e5_interim_and_direction_review.md](14_e3c_e5_interim_and_direction_review.md) | **AARL-full (α=0.5 + BiomedCLIP), 3 seeds:** SLAKE 87.6 (+8.8 over w/o-R), RAD 75.2 (+6.2). English control: vanilla rationales hurt in EN too, 2.5–3× less → **reframe C2**; direction review and missing baselines |
| [12_experiment_progress.md](12_experiment_progress.md) | **Progress chart:** every experiment, run grids incl. E3 BiomedCLIP (SLAKE 88.5 vs 77.9) and E5 English control, critical path (updated 2026-11-22 18:25) |
| [11_v2b_interim_results.md](11_v2b_interim_results.md) | **v2b results (interim):** protocol fixed; SLAKE −13 pts from rationales; α=0.5 segment loss closes the gap (1 seed) |
| [10_v2_week1_sanity.md](10_v2_week1_sanity.md) | Week-1 sanity check: v2 early stopping undertrained runs → protocol changed to v2b |
| [09_status_handoff.md](09_status_handoff.md) | **Start here when resuming:** what is running, next steps, fixes not to regress |
| [08_code_changes_v2.md](08_code_changes_v2.md) | **Implemented** code changes, config switches, week-1 run list, verification status |

## Executive summary

**Where you are:** a careful Bengali replication of MedThink. Its result is negative: every rationale strategy is **worse** than no rationale in Bengali (−3 to −15 accuracy points), the opposite of English. On its own, that is not enough for a Q1 journal.

**What is actually interesting (found in your own logs):**
1. The reversal is statistically significant (McNemar χ² = 15.0, p ≈ 1e-4 on R-SLAKE closed).
2. **Loss dilution.** Answers are ~2% of target tokens. w/o-R training loss reaches ≈ 0, while Explanation plateaus at 1–2.6 because the 247M Bengali decoder cannot model the long rationales.
3. **The model barely looks at the image.** Mean gate λ is 0.05–0.15, barely moved from its 0.047 init.
4. Generic COCO DETR features; rationales contain the answer 38–75% of the time (this explains the Two-Stage collapse).

**Proposed paper:** *"When Rationales Hurt: Diagnosing and Repairing Rationale-Supervised Medical VQA for Low-Resource Bangla"*
- C1: human-verified, EN–BN parallel, rationale-annotated benchmark (first of its kind)
- C2: controlled cross-lingual evidence of the reversal (same-pipeline English control)
- C3: mechanistic diagnosis (loss share, visual under-use, rationale length)
- C4: **AARL**: segment-balanced loss + BiomedCLIP features + gate warm-up (cheap, 0.25B, single T4)
- C5: evaluation suite + clinician rationale ratings + zero-shot VLM baselines

**Budget:** Tier 1 ≈ 85 GPU-h, which fits in ~3 weeks of one Kaggle account (faster with co-author accounts). Decision gate on **Dec 7**.

**This week's to-dos:**
1. Implement the code changes in doc 03 (val split, early stopping, seeds, segment loss, choice scoring, image ablation).
2. Extract BiomedCLIP features.
3. Read the two competing Bangla Med-VQA papers.
4. Request MedGemma access.
5. Check whether raw (pre-edit) LLM translations were kept, for post-edit statistics.
