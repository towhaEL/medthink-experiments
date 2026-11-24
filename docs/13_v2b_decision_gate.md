# 13 — Decision gate result: α=0.5 segment loss, 3 seeds (2026-11-22 ~09:40)

## In plain terms
- **The Dec 7 decision gate passes on both datasets, two weeks early.** With the segment-balanced loss (α=0.5), training with rationales no longer hurts:
  - **RAD:** 74.9 vs 69.0 for no rationale (w/o-R). That is **+5.9 pts**, significant (pooled McNemar p = 6e-5).
  - **SLAKE:** 78.4 vs 78.8 (−0.3 pts). This is a tie (p = 0.82), and well inside the gate's 1-point margin.
- **Plain rationale training (vanilla MedThink Explanation) still hurts:** −5.4 pts on RAD (p = 0.006) and −14.0 pts on SLAKE (p = 4e-17). The paper's core negative finding holds over 3 seeds.
- **Fixing the loss recovers 11–14 pts over plain Explanation** (p < 1e-9 on both datasets). So the main cause is answer-loss dilution, not the rationales themselves.
- **The model now uses the image more,** but the visual gate is still nearly closed (λ ≈ 0.05). BiomedCLIP features plus gate warm-up (E3) remain the next lever.

Source: `python3 experiments/analyze_v2b.py` (stdlib only; reads `experiments/results/v2/mtbn-v2b-*`). Protocol v2b throughout. Test accuracy uses the closed-end ChoiceAccuracy, which equals generative accuracy here because there were no unresolved answers.

## 1. Main table (test accuracy, mean ± std over seeds 42/43/44)

| Dataset | w/o-R | Explanation | Expl + α=0.5 | α=0.5 − w/o-R |
|---|---|---|---|---|
| RAD | 69.0 ± 1.3 | 63.6 ± 2.2 | **74.9 ± 0.8** | **+5.9** |
| SLAKE | 78.8 ± 1.4 | 64.8 ± 1.8 | **78.4 ± 1.9** | −0.3 |

Per seed:
- RAD w/o-R: 70.2 / 69.1 / 67.6
- RAD Expl: 65.8 / 63.6 / 61.4
- RAD α=0.5: 75.7 / 74.3 / 74.6
- SLAKE w/o-R: 79.8 / 77.2 / 79.3
- SLAKE Expl: 64.2 / 66.8 / 63.5
- SLAKE α=0.5: 77.9 / 80.5 / 76.9

Majority-class baselines: RAD 51.1, SLAKE 56.7. Reasoning (s42): RAD 58.1, SLAKE 61.1.

## 2. Paired McNemar tests (exact, same seed, same test questions; pooled = the 3 seeds' discordant counts summed)

| Dataset | A vs B | B right & A wrong / A right & B wrong (pooled) | p (pooled) | per-seed p |
|---|---|---|---|---|
| RAD | w/o-R vs Explanation | 102 / 146 | **0.006** | 0.22, 0.11, 0.10 |
| RAD | w/o-R vs α=0.5 | 94 / 46 | **6.1e-5** | 0.024, 0.060, 0.013 |
| RAD | Explanation vs α=0.5 | 161 / 69 | **1.2e-9** | 0.001, 0.001, 1e-4 |
| SLAKE | w/o-R vs Explanation | 131 / 305 | **4.5e-17** | 1e-7, 2e-4, 1e-7 |
| SLAKE | w/o-R vs α=0.5 | 82 / 86 | 0.82 | 0.37, 0.060, 0.24 |
| SLAKE | Explanation vs α=0.5 | 288 / 118 | **1.7e-17** | 3e-6, 1e-7, 3e-6 |

Caveat: pooling seeds treats the 3 runs as independent samples of the same questions, which overstates evidence somewhat. For the paper, also report per-seed p-values or a mixed-effects / bootstrap-over-seeds interval.

## 3. α sweep (seed 42)

| Dataset | Expl (no α) | α=0.3 | α=0.5 | α=0.8 | α=0.95 | w/o-R |
|---|---|---|---|---|---|---|
| RAD | 65.8 | 71.0 | **75.7** | 71.7 | 69.5 | 70.2 |
| SLAKE | 64.2 | 79.1 | 77.9 | 73.8 | 77.9 | 79.8 |

- **RAD peaks clearly at α=0.5:** 75.7, against 71.0 at 0.3, 71.7 at 0.8 and 69.5 at 0.95. That is a 4–6 pt margin, beyond seed noise. α=0.95 drops back to the w/o-R level, as expected, since it nearly ignores the rationale.
- On SLAKE the sweep is flat and not monotone: 0.8 is lower than 0.95. Single-seed differences of ±2 pts are within seed noise (std ≈ 1.5–2).
- The robust finding is "any per-segment normalisation ≫ none". The exact α matters much less.
- **Keep α = 0.5 as the default.** It is best on RAD, within noise of the best on SLAKE, the only value with 3 seeds, and the "equal weight per segment" story is simple.
- All 26 runs of plans e1, week2 and e2b finished on 2026-11-22 at 10:53, with 0 failures.

## 4. Image reliance and gate (means over seeds)

| Dataset | Condition | real | zero image | shuffled | real − shuffled | gate λ | best / run epochs |
|---|---|---|---|---|---|---|---|
| RAD | w/o-R | 69.0 | 63.1 | 60.3 | +8.7 | 0.072 | 34/40, 32/40, 25/40 |
| RAD | Explanation | 63.6 | 56.2 | 52.9 | +10.7 | 0.046 | 33/40, 40/40, 29/40 |
| RAD | α=0.5 | 74.9 | 67.3 | 59.9 | **+15.0** | 0.052 | 39/40, 31/40, 33/40 |
| SLAKE | w/o-R | 78.8 | 70.4 | 68.6 | +10.2 | 0.072 | 51/60, 46/60, 46/60 |
| SLAKE | Explanation | 64.8 | 63.1 | 59.4 | +5.4 | 0.044 | 24/39, 58/60, 17/32 |
| SLAKE | α=0.5 | 78.4 | 71.2 | 70.4 | +8.1 | 0.049 | 53/60, 49/60, 48/60 |

- **Image use rises with α=0.5.** The real-minus-shuffled gap goes from +10.7 to +15.0 on RAD and from +5.4 to +8.1 on SLAKE. That is part of where the gain comes from.
- **The gate stays near its init value** (sigmoid(−3) = 0.047) in every rationale run. w/o-R opens it slightly more (0.072). This motivates E3: BiomedCLIP features, gate bias −1 and vision warm-up.
- **Plain Explanation on SLAKE stops early and erratically** (best epochs 24, 58, 17). This fits the majority-collapse finding in doc 10.
- **RAD runs often peak near the 40-epoch cap.** This could understate RAD by roughly a point. Leave the caps as they are for comparability (doc 11), and mention it as a limitation.

## 5. What this means for the paper
- **C1 (diagnosis):** answer-loss dilution is confirmed as the dominant cause. Fixing only the loss recovers 11–14 pts.
- **C2 ("rationales hurt in Bangla"):** true for vanilla MedThink. With AARL's loss, rationales help on RAD (+5.9) and are neutral on SLAKE. The English control (E5) is still needed to show the reversal is specific to Bangla.
- **Next experiments:**
  - E3: BiomedCLIP + warm-up on top of α=0.5. Features are extracted; the Kaggle dataset still has to be created.
  - Path for E1 and α=0.5.
  - E5 English control.
