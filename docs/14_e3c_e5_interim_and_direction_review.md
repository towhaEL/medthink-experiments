# 14 — E3c / E5 / Path interim results + research-direction review (2026-11-24)

Source: `python3 experiments/analyze_conditions.py` (new; groups every v2b run by condition over seeds, plus pooled McNemar). Protocol v2b throughout. Final tables wait for E3c w/o-R + BMC, E5b and Path.

## In plain terms
- **AARL-full (α=0.5 + BiomedCLIP) is confirmed over 3 seeds.**
  - SLAKE: **87.6 ± 1.0** vs w/o-R (DETR) 78.8 → **+8.8**, p = 1e-13.
  - RAD: **75.2 ± 0.4** vs 69.0 → **+6.2**, p = 1e-4.
- **On SLAKE the model now really uses the image:** zero-image accuracy falls to 61.0 and the real − shuffled gap is +22.6 (DETR α=0.5: +8.1).
- **The English control changes the story.** Plain rationale training hurts in English too, just less:

| Δ vs w/o-R (same language) | RAD BN (3 seeds) | RAD EN (1 seed) | SLAKE BN (3 seeds) | SLAKE EN (1 seed) |
|---|---|---|---|---|
| Vanilla Explanation | **−5.4** | −1.9 (p=0.62) | **−14.0** | −5.7 (p=0.025) |
| Explanation + α=0.5 | **+5.9** | **+6.9** (p=0.005) | −0.3 | 0.0 |

  So "rationales help in English but hurt in Bangla" is **not** what we observe. What we observe: under a multi-seed, early-stopped protocol, plain rationale supervision hurts in both languages, the damage is about **2.5–3× larger in Bangla**, and segment balancing removes it in both. E5b (EN seeds 43/44, 12 runs) was queued today to make this a 3-seed claim.
- **Rationale quality is not traded away:** at α=0.5 ROUGE-L is 19.7 vs 20.6 on RAD and 23.2 vs 23.0 on SLAKE. α≥0.8 does degrade rationales (RAD α=0.95: ROUGE-L 9.9).
- **Path (partial):** w/o-R 82.9 (2 seeds), Explanation 73.0, α=0.5 82.5 (1 seed each). With DETR features, Path ignores the image completely (zero image = real). Path + BiomedCLIP is needed.

## 1. Condition table (test accuracy)

| Dataset | Condition | n | Acc | zero img | shuffled | real − shuf | λ |
|---|---|---|---|---|---|---|---|
| RAD | w/o-R | 3 | 69.0 ± 1.3 | 63.1 | 60.3 | +8.7 | 0.072 |
| RAD | w/o-R + BMC | 1 (s43) | 68.8 | 50.7 | 46.0 | +22.8 | 0.053 |
| RAD | Explanation | 3 | 63.6 ± 2.2 | 56.2 | 52.9 | +10.7 | 0.046 |
| RAD | α=0.5 | 3 | 74.9 ± 0.8 | 67.3 | 59.9 | +15.0 | 0.052 |
| RAD | **α=0.5 + BMC (AARL-full)** | 3 | **75.2 ± 0.4** | 63.7 | 59.8 | +15.4 | 0.030 |
| SLAKE | w/o-R | 3 | 78.8 ± 1.4 | 70.4 | 68.6 | +10.2 | 0.072 |
| SLAKE | Explanation | 3 | 64.8 ± 1.8 | 63.1 | 59.4 | +5.4 | 0.044 |
| SLAKE | α=0.5 | 3 | 78.4 ± 1.9 | 71.2 | 70.4 | +8.1 | 0.049 |
| SLAKE | **α=0.5 + BMC (AARL-full)** | 3 | **87.6 ± 1.0** | 61.0 | 65.0 | +22.6 | 0.032 |
| RAD EN | w/o-R / Expl / α=0.5 | 1 each | 74.3 / 72.4 / 81.2 | | | | |
| SLAKE EN | w/o-R / Expl / α=0.5 | 1 each | 80.5 / 74.8 / 80.5 | | | | |
| Path | w/o-R / Expl / α=0.5 | 2 / 1 / 1 | 82.9 / 73.0 / 82.5 | ≈ real | ≈ real | ≈ 0 | |

Still running (E3c): w/o-R + BMC (RAD s42/s44, SLAKE ×3), vanilla Expl + BMC s42 (RAD, SLAKE); Path seeds; E5b.

**The key pending control is SLAKE w/o-R + BMC.** If it also reaches ~87, the SLAKE gain comes from the encoder, and the rationale's contribution is "no cost" rather than "gain". RAD w/o-R + BMC s43 (68.8, no gain) already suggests that on RAD the +6 comes from the segment-balanced rationale loss, not from the features.

## 2. Research-direction review

### Verdict: keep the direction, change the headline claim
The work is in good shape and ahead of the timeline (the Dec 7 gate passed on Nov 22). The core contributions stand: a rationale-annotated, human-checked Bangla Med-VQA benchmark; a controlled diagnosis; a simple fix with large, significant gains; a 0.25B model on a free GPU. Four things should change.

### Change 1 — reframe C2 (needed; the data contradicts the current framing)
- Old: *"rationale supervision helps in English but reverses in Bangla."*
- New: *"Standard rationale supervision does not reliably help small Med-VQA models even in English once evaluated over seeds with early stopping (MedThink's +4.5/+3.8 does not reproduce), and it hurts much more in a low-resource language (2.5–3×). The cause is answer-loss dilution, and balancing the loss turns rationales into a gain in both languages."*
- This is a stronger and more honest paper. A non-replication with a mechanism and a fix is publishable. Claiming an English benefit that our own control does not show would be caught by reviewers.
- Suggested title: *"When Rationales Hurt: Answer-Loss Dilution in Rationale-Supervised Medical VQA, and Its Amplification in Low-Resource Bangla"*.
- Needs E5b (3 EN seeds). Check whether the language × condition interaction is significant (e.g. a logistic mixed model or a bootstrap over seeds of Δ_BN − Δ_EN).

### Change 2 — present AARL as a diagnosis-driven fix, not a novel loss
- Per-segment weighting of answer and rationale losses is close to *Distilling step-by-step* (Hsieh et al., 2023), which weights label and rationale losses separately. A reviewer will say so.
- So the novelty claim should be: the **diagnosis** (answer ≈ 2% of target tokens → dilution, worse with longer Bangla targets), the **controlled cross-lingual evidence**, and showing that a medical encoder only pays off once the loss is fixed (SLAKE: +10 pts, with real image dependence). Cite Hsieh et al. directly as the source of the weighting.
- Gate warm-up has been dropped, so AARL = segment loss + BiomedCLIP. Update doc 04 wording.

### Change 3 — missing baselines that reviewers will ask for
| Missing | Why | Cost |
|---|---|---|
| **w/o-R + BMC** (running) | Separates encoder gain from rationale gain | in queue |
| **Rationale dropout 50%** (doc 04 calls it "required"; not implemented) | The obvious simple alternative to segment weighting | ~1 day code + 6 runs |
| **Zero-shot / fine-tuned VLM rows (E6)** | Q1 reviewers expect at least Qwen2.5-VL-3B and a medical VLM | notebook not built |
| **Path + BMC** | Path currently ignores the image; without it Path is a weak third dataset | 6 runs |
| **Open-end results (E8)** | MedThink reports open-end; closed-only looks partial | template not built |

### Change 4 — the non-compute work is now the critical path
Compute is ahead of plan; these are behind and decide Q1 vs Q2:
- **Human evaluation of rationales (doc 06).** Clinicians need sheets *before* the holidays (W4 = Dec 15–21). Outputs for vanilla Expl and AARL-full exist now, so the sheets can be built this week.
- **Read the two Bangla Med-VQA competitor papers (doc 02).** Not done yet. The ACL Findings 2026 paper is a direct competitor; the differentiation paragraph depends on it.
- **Translation-quality statistics for C1** (doc 06-A).

### Things to state as limitations (new)
- RAD: 97% of test images also appear in train, so BMC shows no gain there (memorisation works with any features). SLAKE's unseen test images are the real test of visual grounding.
- The gate λ mean is a poor proxy for image use (SLAKE AARL-full: λ 0.032, yet −27 pts with a zero image). Use the ablations.
- BN rationale n-gram scores are not comparable with EN ones (morphology, tokenisation).

## 3. Recommended next steps (in order)
1. When E3c finishes: fetch, re-run `analyze_conditions.py`, and decide the SLAKE framing (gain vs "no cost").
2. Path + BMC (w/o-R, Expl, α=0.5 × 3 seeds) — queue once w/o-R + BMC lands.
3. Implement rationale dropout (sanity run first), run on RAD + SLAKE with BMC.
4. Build human-eval sheets from vanilla Expl vs AARL-full outputs; send to clinicians by ~Dec 5.
5. Build E6 (Qwen2.5-VL-3B zero-shot; MedGemma if HF access is granted) and E8 open-end.
6. Read competitor papers; rewrite doc 04's contribution list with the new C2.

---

## 4. Update 2026-11-24 12:35 — w/o-R + BMC landed, more EN seeds

### In plain terms
- **SLAKE: the gain comes from BiomedCLIP, not from the rationales.** w/o-R + BMC reaches **86.7 ± 1.8**, a tie with AARL-full's 87.6 (p = 0.32). With BiomedCLIP, rationales on SLAKE are "explanations at no accuracy cost" (+0.9, n.s.).
- **RAD: the gain comes from the rationales (with the balanced loss), not from BiomedCLIP.** w/o-R + BMC is 70.1 (vs 69.0 with DETR, p = 0.59), and AARL-full is **+5.1 over it** (75.2, p = 0.002).
- **BiomedCLIP does not rescue plain rationale training:** RAD vanilla Expl + BMC is 64.0 (1 seed), the same as vanilla Expl with DETR (63.6), and 11 pts under α=0.5 + BMC (p = 9e-4). SLAKE Expl + BMC is still running and decides whether this holds on SLAKE too.
- **English RAD now has 3 w/o-R seeds and 2 Explanation seeds:** 75.9 vs 71.3 → **Δ −4.6 in English vs −5.4 in Bangla.** On RAD, plain rationale training hurts about as much in English as in Bangla. The "2.5–3× worse in Bangla" estimate from §2 rested on single EN seeds and does not survive on RAD. SLAKE EN is still 1 seed (−5.7 vs −14.0).
- **Language gap in the baseline:** w/o-R is 6.9 pts lower in Bangla than in English on RAD (69.0 vs 75.9). Bangla AARL-full (75.2) roughly closes it, but English also gains from α=0.5 (81.2, 1 seed).
- **Path w/o-R: 83.0 ± 0.5 (3 seeds)**, with no image use (zero image 82.5). Path + BMC (AARL-full s42, then w/o-R + BMC s42) launched on the free `mdsafiulalamsarker` slot (plan `pathbmc`).

### Effect decomposition (3 seeds, pooled McNemar)
| | RAD | SLAKE |
|---|---|---|
| BiomedCLIP alone (w/o-R → w/o-R + BMC) | +1.1 (p = 0.59) | **+7.9** (p = 3e-11) |
| Rationales + α=0.5 on top of BMC (w/o-R + BMC → AARL-full) | **+5.1** (p = 0.002) | +0.9 (p = 0.32) |
| Total (w/o-R → AARL-full) | **+6.2** (p = 1e-4) | **+8.8** (p = 1e-13) |

Why the two datasets split: RAD test images are 97% seen in training, so better features add little and the rationale's extra supervision helps. SLAKE test images are unseen, so generalising visual features are what matters.

### What this changes in the paper framing
- **C4 claim:** "AARL improves accuracy" is true overall, but the parts act on different datasets. State it as a decomposition, not as one method that helps everywhere.
- **C2 claim:** do **not** claim the harm is clearly amplified in Bangla until E5b finishes. Current evidence: harmful in both languages; similar on RAD; possibly larger in Bangla on SLAKE. A safer headline is the non-replication plus the dilution mechanism (true in both languages) plus the fix. Treat Bangla amplification as a secondary finding, conditional on E5b SLAKE.
- **Missing baseline, now more important:** rationale dropout. Also report w/o-R + BMC as the strongest no-rationale baseline in every table.

---

## 5. Update 2026-11-24 14:15

### In plain terms
- **A medical encoder does not rescue plain rationale training.** Vanilla Explanation + BMC (s42): SLAKE **68.8** vs AARL-full 87.6 (p = 7e-16); RAD 64.0 vs 75.2 (p = 9e-4). BMC only pays off when the answer loss is not diluted (w/o-R + BMC 86.7, α=0.5 + BMC 87.6). This supports the dilution mechanism. Seed 43 is running (plan `e3d`).
- **English control, more seeds:**

| Δ vs w/o-R (same language) | RAD BN | RAD EN | SLAKE BN | SLAKE EN |
|---|---|---|---|---|
| w/o-R level | 69.0 (3) | 75.9 (3) | 78.8 (3) | 81.4 (3) |
| Vanilla Explanation | −5.4 (3) | **−4.3** (3, p = 0.015) | −14.0 (3) | −6.6 (1) |
| Explanation + α=0.5 | +5.9 (3) | +4.2 (2, p = 0.003) | −0.3 (3) | −0.4 (2) |

  - The harm is **clear and significant in English** too. On RAD the two languages behave the same; on SLAKE the Bangla drop is about twice as large (EN Expl still 1 seed; s43/s44 running).
  - The fix works the same way in both languages.
  - **Implication:** the paper's main claim is that rationale supervision as practised hurts small Med-VQA models in both languages, because of answer-loss dilution. Bangla amplification is SLAKE-only so far.
- **Path:** vanilla Explanation 73.5 ± 0.7 (2 seeds) vs w/o-R 83.0 (p = 5e-67); α=0.5 82.5 (1 seed). Path + BMC (AARL-full and w/o-R, s42) are running.

### Slot use at 14:13
- **towhaelahi:** EN RAD α=0.5 s43 + EN SLAKE α=0.5 s43 (moved here from the full nishatibnatmaimuna).
- **codegiveway:** Path Expl s44 + SLAKE Expl + BMC s43; RAD Expl + BMC s43 queued.
- **sefatahmedakkhor / nishatibnatmaimuna:** Path α=0.5 s43/s44 + EN SLAKE Expl s43/s44.
- **mdsafiulalamsarker:** Path + BMC (AARL-full, w/o-R) s42.
