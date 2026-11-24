# 10 — Week-1 v2 sanity check (2026-11-21 ~13:00)

8 of the 14 week-1 runs were finished and downloaded, results files only, with `experiments/fetch_results.py` into `experiments/results/v2/<slug>/`.

## Verdict: the runs are technically clean but undertrained
- No NaN batches, no divergence. Every run trained past `min_epochs`.
- The v2 early-stopping rule halts training while validation accuracy is still rising, so test accuracy sits **15–17 pts below** the old paper-protocol runs.

| Run | Epochs (best) | Test ChoiceAcc | Majority | Paper-protocol run (old) |
|---|---|---|---|---|
| RAD Expl s42/43/44 | 14(9) / 11(6) / 17(12) | 54.8 / 56.6 / 58.1 → **56.5** | 51.1 | 71.7 (150 ep) |
| RAD w/o-R s42 | 10(4) | **57.4** | 51.1 | 75.0 (150 ep) |
| SLAKE Expl s42/43/44 | 20(15) / 15(3) / 22(17) | 62.3 / 56.7 / 61.8 → **60.3** | 56.7 | 73.8 (300 ep) |
| SLAKE w/o-R s42 | 17(12) | **70.7** | 56.7 | 82.2 (300 ep) |

### Validation curves (ChoiceAccuracy per epoch)
```
slake-wor-s42   41 62 62 61 62 62 64 68 72 72 69 75 73 73 73 74 75   <- still climbing when stopped
rad-expl-s44    48 53 48 54 55 61 63 57 57 59 64 66 63 63 64 65 64
slake-expl-s43  50 38 62 62 45 62 62 62 62 62 62 62 61 62 60         <- stuck at the majority answer
```
- **Cause.** Validation is small and noisy (RAD has about 180 val items, so one item is about 0.5 pt). With patience 5, a single lucky epoch triggers the stop.
- **Learning rate.** It decays over the epoch cap (40/60), so the runs stop while the LR is still high.

## Other observations (useful for the paper)
1. **SLAKE Explanation runs plateau at the majority answer** (val ≈ 62 for most epochs). s43 is a full collapse: test accuracy equals the majority baseline.
   - This fits the answer-loss dilution hypothesis, because the rationale tokens dominate the loss.
2. **The visual gate never opens.** *(Correction, doc 11: the shuffle result below was an undertraining artifact. Under v2b, shuffling the images costs 8–13 pts.)*
   - λ_mean is 0.036–0.077 against an init of 0.047 (`gate_bias_init=-3`), and `shift_from_init` ≈ −0.01.
   - Image ablation: shuffled-image accuracy ≈ real-image accuracy on SLAKE w/o-R (70.0 vs 70.7). The model barely uses the image.
3. **The model overfits the training set.** Train `ans_loss` ends at 0.15–0.58.

## Cost
Measured on T4: about 0.033 GPU-h per epoch with rationale and about 0.011 without.
- Running to the full caps (40 RAD / 60 SLAKE) costs at most 1.3 h / 2.0 h per Explanation run.
- Re-running all 12 E1 seed runs this way costs about 15–18 GPU-h in total. That is well inside the 60 h/week across both accounts.

## Options (decision needed before week 2)
- **A. Recommended: loosen the stopping rule and re-run E1.**
  - Settings: patience 15/15/5 (RAD/SLAKE/Path), `min_epochs` = 50% of cap, caps unchanged.
  - Selection still uses validation, so the protocol stays principled, and the runs get close to convergence.
- **B. Fixed-length training.** Train to the full cap and keep the best validation checkpoint. This is simplest and costs slightly more.
- **C. Keep the current protocol.** The w/o-R vs Expl comparison is still internally consistent. However, baselines 15 pts below our own earlier numbers will draw reviewer objections.

Whatever we choose, the same protocol must be used for E2 (`ans_weight`) and everything after it.
