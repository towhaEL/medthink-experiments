# 11 — v2b results, interim (2026-11-21 ~19:50)

**Status:** 13 of 20 v2b runs finished; the results are in `experiments/results/v2/mtbn-v2b-*/`.

**Still running (pushed 19:40, about 2 h):**
- SLAKE Explanation s42 at α=0.95
- RAD w/o-R s42–s44
- RAD Explanation s42 at α 0.5, 0.8 and 0.95

**Scheduler stall:** the scheduler stopped polling after 14:56 for about 5 h. The likely cause is a host suspend, which pauses Python's sleep timer. It was restarted at 19:39 and nothing was lost.

## 1. The v2b protocol fixed the undertraining
- Validation curves are now flat at the end (see the last 8 epochs in the table below).
- No NaNs, no divergence.
- SLAKE w/o-R now averages **78.8 ± 1.4**, close to the old paper-protocol 82.2. Under v2 it was 70.7.
- Many runs reach the epoch cap: RAD Expl s43 has its best epoch at 40/40, SLAKE Expl s43 at 58/60.
  - Their curves are flat, so the cap costs at most ~1 pt.
  - Keep the caps as they are, so all runs stay comparable.

| Run (closed-end, test ChoiceAcc) | s42 | s43 | s44 | Mean ± sd | Old v2 (patience 5) | Paper protocol |
|---|---|---|---|---|---|---|
| RAD Explanation | 65.8 | 63.6 | 61.4 | **63.6 ± 2.2** | 56.5 | 71.7 |
| RAD w/o-R | running | running | running | — | 57.4 | 75.0 |
| SLAKE Explanation | 64.2 | 66.8 | running | **65.5** (n=2) | 60.3 | 73.8 |
| SLAKE w/o-R | 79.8 | 77.2 | 79.3 | **78.8 ± 1.4** | 70.7 | 82.2 |
| RAD Reasoning (`Accuracy`) | 58.1 | — | — | — | — | 65.1 |
| SLAKE Reasoning (`Accuracy`) | 61.1 | — | — | — | — | 68.5 |

Majority-class baselines: RAD 51.1, SLAKE 56.7.

**The core finding replicates under a clean protocol.** On SLAKE, rationale supervision costs **−13.3 pts** (Explanation 65.5 vs w/o-R 78.8).

## 2. E2 first readout: the segment-balanced loss closes the gap on SLAKE
SLAKE Explanation, seed 42:

| `ans_weight` α | Test ChoiceAcc | Best val | Rationale ROUGE-L | chrF |
|---|---|---|---|---|
| None (paper token-mean) | 64.2 | 67.4 | 22.9 | 30.4 |
| **0.5** | **77.9** | 82.9 | **23.2** | 30.6 |
| 0.8 | 73.8 | 80.8 | 20.8 | 25.9 |
| 0.95 | running | | | |
| *w/o-R reference (3-seed mean)* | *78.8* | | | |

- **α = 0.5 gains +13.7 pts over vanilla Explanation, which brings it to 0.9 pts below the w/o-R mean.**
  - That is inside the decision-gate threshold (≥ w/o-R − 1 pt) on SLAKE.
  - Rationale quality is not hurt: ROUGE-L 23.2 vs 22.9.
- **α = 0.8 is worse on both answers and rationales.**
  - So the gain does not come from "more answer weight". It comes from **length-normalising each segment**: equal weight for the answer and the rationale, instead of the answer's ~2% share of tokens.
  - That is a cleaner story for the paper.
- **Caveat: this is a single seed.** Before Dec 7, run α=0.5 on seeds 43 and 44, and on RAD. If α=0.95 or a value below 0.5 (e.g. 0.3) beats 0.5, include it as well.

## 3. Correction to doc 10: the image *is* used
- Doc 10 reported shuffled ≈ real accuracy. That was an artifact of undertraining.
- Under v2b, shuffling the images costs 8–13 pts on SLAKE: w/o-R 79.8 → 66.8, α=0.5 77.9 → 71.6.
- The visual gate still stays near its initial value (λ ≈ 0.04–0.075 vs 0.047 at init). The model uses the image through a nearly closed gate.
  - Gate warm-up and BiomedCLIP are still worth testing, but they are no longer a rescue.

## 4. Next steps
1. When the remaining 7 runs finish (around 21:45):
   - `python3 experiments/fetch_results.py --plan e1` and `--plan week2`
   - complete the table above with RAD w/o-R and the RAD α sweep
2. **E2b (queued 19:57, plan `e2b`):** α=0.5 Explanation for seeds 43/44 on RAD and SLAKE, plus α=0.3 at seed 42.
3. Add the v2b aggregation to notebook 05 section 1b (mean ± sd, McNemar test w/o-R vs α=0.5).
