# 09 — Status & Handoff (snapshot 2026-11-24 ~15:55)

Read this first when resuming. Full background: docs 01–08 (index in `README.md`). Latest results: doc 13 (decision gate), doc 12 (progress chart and run grids).

## Current state in one glance (2026-11-22 18:25)
- **Decision gate passed** (doc 13). α=0.5 segment loss: RAD +5.9 over w/o-R; SLAKE tie.
- **BiomedCLIP (E3b): SLAKE 88.5 vs DETR 77.9** (1 seed, p = 1e-8). The model now really uses the image. RAD shows no gain. Gate warm-up is dropped.
- **English control (E5), partial:** RAD Δ(Expl − w/o-R) is EN −1.9 vs BN −4.4. Rationales do not *help* in English on RAD. SLAKE EN Expl is pending and decides how C2 is phrased.
- **Path, partial:** w/o-R 83.4, Expl 73.0. DETR images are ignored on Path (zero image = real image).
- **Running:** a single scheduler, `--plan e3b,pathsanity,e5,e3c,pathe1`, PID in `experiments/e3_scheduler.pid`, log `experiments/e3_scheduler.log`. 10 slots busy: 3 E5, 5 E3c, 2 Path. 7 E3c and 5 Path runs queued.
- **Fetch / analyse:** `python3 experiments/fetch_results.py --plan e3b,e5,e3c,pathe1`, then `python3 experiments/analyze_v2b.py`. The script covers E1/E2 only; extend it for E3c/E5/Path when writing doc 14.

## Research in one paragraph
Bengali Med-VQA built on MedThink (Findings NAACL 2025). Rationale supervision **helps in English but hurts in Bangla** (−3 to −15 pts, McNemar p≈1e-4 on R-SLAKE). Diagnosed causes:
- **answer-loss dilution:** the answer is ~2% of target tokens
- **near-closed visual gate:** λ 0.05–0.15
- **generic COCO DETR features**

Planned fix **AARL** = segment-balanced loss (`ans_weight`) + BiomedCLIP features + gate warm-up. Target: Q1 journal (Computers in Biology and Medicine / AI in Medicine), submit ~Jan 12–15 2027. **Decision gate Dec 7**: does best-α Explanation reach ≥ w/o-R − 1 pt on RAD and SLAKE?

## Constraints (from the user)
- Kaggle T4 only, 30 GPU-h/week per account, 2 concurrent runs each. **5 accounts** (managed by **KaggleFleet**, all with access to both datasets): `towhaelahi` (owner), `codegiveway`, `sefatahmedakkhor`, `mdsafiulalamsarker`, `nishatibnatmaimuna` (co-authors) → 10 parallel runs, 150 GPU-h/week.
- Dataset is LLM-translated and **fully human-checked**. 2+ Bengali-speaking clinicians are available for evaluation.
- All outputs go to `docs/` as Markdown.

## What happened on 2026-11-21 (midday)
- **Week-1 sanity check: the runs were undertrained** (details in doc 10).
  - Under v2 (patience 5, `min_epochs` 25%), early stopping fired while val accuracy was still rising.
  - Test accuracy came out 15–17 pts below the old paper-protocol runs: SLAKE w/o-R 70.7 vs 82.2, RAD Expl 56.5 vs 71.7.
  - Otherwise clean: no NaNs, no divergence.
- **The user chose to loosen early stopping and re-run E1.** This is protocol **v2b**: patience 15/15/5 and `min_epochs` = 50% of cap.
  - Library defaults have been changed and the notebooks rebuilt.
  - New kernels use the prefix `mtbn-v2b-`. The old `mtbn-v2-` week-1 runs are superseded; never mix them with v2b numbers.
- **Downloaded:** results for 8 old week-1 runs, in `experiments/results/v2/mtbn-v2-*/`.
- **Findings worth keeping for the paper:**
  - SLAKE Explanation collapses to the majority answer (s43 does so completely).
  - The visual gate never opens: λ stays at its init value of 0.047.
  - Shuffled-image accuracy ≈ real-image accuracy.

## Scheduler state / what to run
- **Old week-1 scheduler:** the user is stopping it (`pkill -f "launch_runs.py --plan week1 --schedule"`).
  - Its 4 in-flight w/o-R kernels finish on Kaggle. Their results are informational only.
- **New scheduler** (E1 re-run plus the E2 sweep together, so one process owns both accounts' slots):
  `nohup python3 -u experiments/launch_runs.py --plan e1,week2 --schedule --poll 300 >> experiments/v2b_scheduler.log 2>&1 &`
  - Pushes rejected while old kernels still hold slots are retried automatically.
  - Check it is alive: `pgrep -af "launch_runs.py --plan e1"`.
- **14:10, rebalanced over all 5 accounts** (scheduler restarted). All 10 slots were busy right after; the plan lists the per-account split and estimated GPU-h. Expected wall time is about 3–4 h for all 20 runs.
- **Plans in `experiments/launch_runs.py`:**
  - `e1`: 14 runs (12 seed runs + 2 Reasoning)
  - `week2`: `ans_weight` ∈ {0.5, 0.8, 0.95}, seed 42 (RAD on mdsafiulalamsarker, SLAKE on nishatibnatmaimuna)
  - AARL switches are passed per run (`ans_weight=`, `gate_bias=`, `vision_warmup=`) and are encoded in the slug (`-aw80`, `-gb1`, `-vw1`).
- **Estimated cost:** E1 about 18–20 GPU-h plus E2 about 8 GPU-h, split across both accounts (60 h/week).
- **Download results:** `python3 experiments/fetch_results.py --plan e1` (and `--plan week2`).
  - It fetches json/jsonl/log only and skips runs already downloaded.

## Update 2026-11-21 19:50 (see doc 11)
- **v2b status:** 13/20 runs done; 7 re-pushed at 19:40 after a scheduler stall (likely host suspend). Restarting the scheduler is the fix, and it is safe.
- **SLAKE results:**
  - w/o-R 78.8 ± 1.4 vs Explanation 65.5
  - α=0.5 segment loss: **77.9** (1 seed), within the decision gate
- **19:57: E2b queued and partly running** (plan `e2b` in `launch_runs.py`: α=0.5 s43/s44 on RAD and SLAKE, plus α=0.3 s42 on both).
  - The scheduler now runs `--plan e1,week2,e2b`, log `experiments/v2b_scheduler.log`. Restart with the same command if it stalls.
  - Expected done around 00:00.
- **Next:**
  - `fetch_results.py --plan e1`, `--plan week2` and `--plan e2b`
  - 3-seed α=0.5 vs w/o-R table plus McNemar test, which answers the Dec 7 gate

## 2026-11-21 21:09: accounts moved to KaggleFleet
- **Accounts and job history now live in KaggleFleet:** `/home/tn-99778/Towha/kaggle-fleet` (package `kaggle_fleet`, state `~/.kaggle_fleet`).
  - All 5 accounts and `jobs.json` were copied from `~/.kaggle_hub`, and the sandbox links were repointed. `kfleet doctor` passes, and all 5 accounts test valid.
  - `experiments/launch_runs.py` and `fetch_results.py` import `kaggle_fleet`: an installed copy, else `../kaggle-fleet/src`.
- **The project's `kaggle_hub/` and `~/.kaggle_hub` are deprecated.** `~/.kaggle_hub` stays as a backup only.
  - Do not use both tools: OAuth tokens refresh only in the copy that is used.
- **CLI without installing:** `PYTHONPATH=../kaggle-fleet/src python3 -m kaggle_fleet account list`. Alternatively, install it (`pipx install -e ../kaggle-fleet`) to get `kfleet`.
  - Web panel: `kfleet serve` replaces `./kaggle_hub/run.sh`.
- The scheduler was restarted on KaggleFleet: `--plan e1,week2,e2b`, same log.

## Update 2026-11-21 21:45
- Scheduler alive (`--plan e1,week2,e2b`, KaggleFleet). Fetched RAD α=0.5 s43/s44.
- **RAD α=0.5: 74.3 / 74.6** vs RAD Explanation 63.6 ± 2.2 and RAD w/o-R s42 70.2 → segment loss *beats* w/o-R on RAD (so far 1 w/o-R seed).
- SLAKE Explanation s44: 63.5 (3-seed mean 64.8).
- Caveat: RAD runs (cap 40) mostly peak at epoch 29–40, so the cap may still bind for RAD.
- Still running/queued (11): RAD w/o-R s43/s44, RAD α 0.5 s42 / 0.8 / 0.95 / 0.3, SLAKE α 0.95 / 0.3, SLAKE α=0.5 s43/s44.

## 2026-11-21 21:55: notebook 01b (BiomedCLIP) built
- `notebooks/01b_extract_biomedclip_features.ipynb` (from `build_notebooks.py`): open_clip BiomedCLIP ViT-B/16, `forward_features` → (197, 768) float16. Outputs `features_biomedclip.npy` + `name_to_row_biomedclip.json`, with different names from DETR so both datasets can be attached together.
- nb 02/03 config gets `FEATURES = "detr" | "biomedclip"`, which sets `patch_num/patch_dim`. `load_features()` picks the file and asserts the shape. Run names gain `_f197x768` only for non-DETR features, so DETR names and behaviour are unchanged (diffed).
- Launcher: `run(..., features="biomedclip")` → slug suffix `-bmc` + extra dataset `towhaelahi/medthink-bn-biomedclip`.
- Pre-change copies: `backup/pre_bmc/`.
- **21:48 pushed** as `towhaelahi/mtbn-01b-biomedclip-features` (KaggleFleet job `job_20261121_214827_mtbn-01b-bio_95e6`). The user creates the dataset from its output in the Kaggle UI.
- **~22:00 done:** 5131/5131 images, 0 missing, (5131, 197, 768) fp16, 1.55 GB, finite, range ±57, mean pairwise CLS cosine 0.42 (not collapsed). The log's `std inf` was an fp16-accumulator print artefact (fixed in the builder). **Waiting on the user:** Output → New Dataset `medthink-bn-biomedclip` + share.
- **To do:** run 01b on Kaggle (attach `towhaelahi/medthink-dataset` + manifests), upload the output as dataset `medthink-bn-biomedclip`, share it with the 4 co-authors, then do a sanity run with `features="biomedclip"` before E3b.

## 2026-11-21 22:50: stalls and lost logins
- **OAuth for `towhaelahi` and `codegiveway` fails** (`kfleet account test`: "Authentication required"), while the other 3 accounts pass. towhaelahi still pushed fine at 21:48. Re-login is needed (`kfleet account login`, done by the user). Possible cause, unverified: several processes refreshing the same OAuth tokens at once (scheduler, `kfleet serve`, CLI polling).
- **7 in-flight kernels report `KernelWorkerStatus.QUEUED` on Kaggle with empty logs.** Five were pushed 19:40–20:08 and have waited about 3 h. Meanwhile slake-s43-aw50 (pushed 19:57) ran and finished. Check the kernel pages in the Kaggle UI.
- SLAKE α=0.5 s43: **80.5**.

## 2026-11-22 09:45: decision gate PASSED (doc 13)
- The 7 runs that were "queued" overnight ran: the host slept, then the scheduler caught up at 09:23. All of E1, week2 and E2b are done except **RAD α=0.95 and α=0.3** (the scheduler pushes them).
- **3-seed α=0.5 vs w/o-R:** RAD 74.9 vs 69.0 (+5.9, McNemar pooled p=6e-5); SLAKE 78.4 vs 78.8 (−0.3, p=0.82). Vanilla Explanation: −5.4 / −14.0.
- Analysis script: `python3 experiments/analyze_v2b.py` (stdlib, prints Markdown).
- **10:53: all 26 runs done, 0 failed; the scheduler exited by itself.** RAD α sweep: 0.3 → 71.0, 0.5 → 75.7, 0.8 → 71.7, 0.95 → 69.5, so α=0.5 is the default. No scheduler is running now.
- **Next:** E3 (BiomedCLIP; waits for the user's dataset `medthink-bn-biomedclip`), Path E1 + α=0.5, E5 English control.

## 2026-11-22 ~11:45: BiomedCLIP sanity passed
- The user created and shared dataset `towhaelahi/medthink-bn-biomedclip`.
- Plan `bmcsanity` (RAD, `PRESET="sanity"`, 200 examples) ran on co-author accounts, which also proves the share works:

| Check | (a) BMC only (mdsafiulalamsarker) | (b) BMC + gate −1 + warm-up (sefatahmedakkhor) |
|---|---|---|
| Features | (5131, 197, 768) loaded, 0 missing | same |
| Params | 251.7M (DETR 251.3M; +0.4M = 768-dim projection), lm_head untied | same |
| Epoch-1 loss | 18.8 (DETR full run: 17.7) | 33.6 (epoch 1 = frozen-backbone warm-up, expected) |
| Final train loss | 0.17, no NaN | 0.14, no NaN |
| Gate λ | 0.047 → 0.054 (max 0.41) | 0.269 → 0.236 (max 0.86): stays open |
| Run name | `..._f197x768_SANITY` | `..._gb-1_vw1_f197x768_SANITY` |

- Test accuracy on the 100-example sanity split (59 / 47 vs majority 56) is noise. Sanity runs are not results.
- Fixed a cosmetic gate print: it printed "init 0.047" even for gb −1.
- **Plan `e3b` is defined** (6 runs, see doc 12) and waits for a go.
- **Logins:** towhaelahi + codegiveway OAuth broke again. Their credential files have not refreshed since the 23:06/23:12 re-login, while the other 3 refreshed at 09:23. The user must re-login; if it recurs, debug the KaggleFleet refresh.

## 2026-11-22 13:30: E3b running, Path + E5 sanity running
- All 5 logins valid after the user re-logged towhaelahi + codegiveway (tokens 13:05/13:06).
- **Scheduler 2:** `nohup python3 -u experiments/launch_runs.py --plan e3b,pathsanity --schedule --poll 300 >> experiments/e3_scheduler.log 2>&1 &`
  - Pushed all 7 at 13:12: E3b on sefatahmedakkhor, mdsafiulalamsarker, nishatibnatmaimuna; Path sanity on towhaelahi.
  - The old `v2b_scheduler.log` scheduler has exited (all 26 done).
- **E5 English control built** (doc 08): `LANG="en"` swaps the English text into the same manifests and uses UnifiedQA-T5-base.
  - `ensanity` (RAD, sanity preset) pushed by hand on codegiveway at 13:16.
  - Plan `e5`: 6 runs (RAD on codegiveway, SLAKE on towhaelahi).
- **Waiting:** Path sanity → `pathe1` (9 runs); EN sanity → `e5` (6 runs). Check params, initial loss, learning and truncation before each batch.

## 2026-11-22 14:45: sanity passed, E5 + Path launched
- **Path sanity:** 251.3M, answer mask OK, loss 16.8 → 0.23, 61 vs majority 51.
- **EN sanity:** overlay OK on Kaggle, `Answer:` mask OK, 226.6M (UnifiedQA 223M tied + 3.7M visual head), loss 8.35 → 0.004, no NaN.
- **Explanation target length (train, tokens, local check):**
  - BN (BanglaT5): median 112 / 99 / 128 for RAD / SLAKE / Path
  - EN (UnifiedQA): 133 / 122 / 154
  - Nothing over the 320 limit except 0.2% of EN Path.
  - EN targets are ~20% longer, so the answer is an even smaller share of the loss in English (relevant to the dilution argument).
- **Scheduler restarted** (PID in `experiments/e3_scheduler.pid`, log `experiments/e3_scheduler.log`):
  - `--plan e3b,pathsanity,e5,pathe1`
  - All 6 E5 runs pushed at 14:41. The 9 Path runs queue behind them, spread over all 5 accounts.
- **E3b first readout (RAD, s42, α=0.5):**
  - DETR 75.7 (λ 0.053)
  - BiomedCLIP 75.0 (λ 0.038)
  - BiomedCLIP + gate −1 + warm-up 74.3: λ **0.030**, even though it starts at 0.269. Training actively closes the gate.
  - So far BiomedCLIP gives no gain on RAD. SLAKE is pending.
- Lesson: a watcher that runs `pgrep -f <pattern>` matches its own command line, so it never sees the scheduler die. Watch the saved PID with `kill -0` instead.

## 2026-11-22 16:15: E3b SLAKE BiomedCLIP = 88.5
- SLAKE Expl α=0.5 s42 with BiomedCLIP: **88.5** vs DETR 77.9 (+10.6) and w/o-R 79.8.
- Image reliance is now real: zero image 50.2, shuffled 65.9 (DETR: 70.7 / 71.6). Predictions are balanced (218 yes / 198 no). The gain spans organs: lung 82→91, abdomen 74→83, brain 82→95.
- RAD shows no BiomedCLIP gain (75.0 vs 75.7). Likely reason: 97% of RAD test images also occur in train, so any features let the model memorise per image, while SLAKE test images are unseen and need features that generalise. Worth stating in the paper.
- Gate λ mean stays low (0.030), yet zero-image accuracy drops 38 pts. λ mean is not a good proxy for visual reliance; use the ablation.
- **Next:** 3 seeds of BMC α=0.5 on SLAKE + RAD, plus w/o-R + BMC (E3c), after the SLAKE warm-up cells land.

## 2026-11-22 16:20: E3c queued
- SLAKE BMC + gate −1 + warm-up: 85.6 (plain BMC 88.5), so **warm-up is dropped**. AARL-full = α=0.5 + BiomedCLIP.
- Plan `e3c` (12 runs): α=0.5 + BMC s43/s44 (RAD, SLAKE); w/o-R + BMC × 3 seeds (both); vanilla Expl + BMC s42 (both).
- Scheduler restarted, `--plan e3b,pathsanity,e5,e3c,pathe1` (e3c is ahead of Path), PID in `experiments/e3_scheduler.pid`.

## 2026-11-24 10:45: resumed after a 2-day stall (doc 14)
- The scheduler had died on 2026-11-22 at 17:35 (likely host sleep), so nothing was pushed for ~41 h. All 5 logins tested valid; restarted.
- **Scheduler now:** `--plan e3b,pathsanity,e5,e3c,pathe1,e5b`, PID in `experiments/e3_scheduler.pid`, log `experiments/e3_scheduler.log`. 10 slots busy; 15 queued (3 Path, 12 E5b).
- **New plan `e5b`:** EN seeds 43/44 for all 4 E5 conditions (12 runs), queued because the EN vs BN gap is clear.
- **New script:** `python3 experiments/analyze_conditions.py` groups every v2b run by condition (all plans) + pooled McNemar. Use it instead of `analyze_v2b.py` for E3c/E5/Path.
- **Results (doc 14):** AARL-full (α=0.5 + BMC) 3 seeds: SLAKE 87.6 ± 1.0, RAD 75.2 ± 0.4. EN control (1 seed): vanilla Expl −1.9 (RAD) / −5.7 (SLAKE) vs BN −5.4 / −14.0; α=0.5 +6.9 / 0.0 in EN. **C2 must be reframed** ("hurts in both, 2.5–3× more in Bangla"), see doc 14 §2.
- **Pending decision:** SLAKE w/o-R + BMC (running) tells whether the SLAKE gain is from the encoder or from rationales.
- **Next:** doc 14 §3 (Path + BMC, rationale dropout baseline, human-eval sheets, E6, competitor papers).

## 2026-11-24 12:35: w/o-R + BMC in, Path + BMC launched (doc 14 §4)
- **SLAKE w/o-R + BMC 86.7 ≈ AARL-full 87.6 (p=0.32):** the SLAKE gain is from the encoder. **RAD:** BMC adds nothing (70.1); AARL-full +5.1 over w/o-R + BMC (p=0.002).
- EN RAD (3/2 seeds): w/o-R 75.9, Expl 71.3 → Δ −4.6 (BN −5.4). Bangla amplification is not supported on RAD; wait for E5b SLAKE.
- **New plan `pathbmc`** (Path AARL-full s42 + w/o-R + BMC s42 on mdsafiulalamsarker). Scheduler now `--plan e3b,pathsanity,e5,e3c,pathe1,e5b,pathbmc`.
- Still running: SLAKE Expl + BMC s42, Path Expl/α=0.5 seeds, E5b.

## 2026-11-24 14:15 (doc 14 §5)
- Vanilla Expl + BMC s42: SLAKE 68.8, RAD 64.0 → the encoder does not rescue plain rationale training. Plan **`e3d`** (Expl + BMC s43, RAD + SLAKE, codegiveway) added.
- EN 3 seeds: RAD Expl −4.3 (BN −5.4); SLAKE EN Expl s43/s44 running.
- E5b SLAKE α=0.5 s43 moved from nishatibnatmaimuna to towhaelahi (never pushed before the move).
- Scheduler: `--plan e3b,pathsanity,e5,e3c,pathe1,e5b,pathbmc,e3d`. Queue after the current runs: only RAD Expl + BMC s43.

## 2026-11-24 15:55 (doc 14 §6, doc 08 rat_drop)
- 10/10 slots busy: Path + BMC s43/s44 (`pathbmc2`), English Path s42 (`pathen`), Expl + BMC s43 (`e3d`), EN SLAKE α=0.5 s43.
- **Rationale dropout implemented** (`rat_drop`, doc 08). `rdsanity` is queued on towhaelahi. **Check its log** (answer mask, `[rat_drop]` line, params, loss), then add plan `rd` (6 runs) to the scheduler.
- Scheduler: `--plan e3b,pathsanity,e5,e3c,pathe1,e5b,pathbmc,e3d,pathbmc2,pathen,rdsanity`.

## Next steps (in order, updated 2026-11-22)
1. **When the batch finishes:** fetch, extend `analyze_v2b.py` (E3c, E5, Path), and write **doc 14**:
   - the 3-seed AARL-full (α=0.5 + BMC) vs w/o-R and w/o-R + BMC table, with McNemar
   - the EN vs BN Δ table
   - Path
2. **E5 follow-up:** if SLAKE shows a clear EN vs BN gap, add EN seeds 43/44 (8 runs).
3. **Path + BiomedCLIP** (AARL-full on Path), once E3c confirms.
4. **Still to build:**
   - E6 zero-shot VLM notebook (Qwen2.5-VL-3B; MedGemma needs HF access)
   - E4 rationale-length flag
   - E8 open-end launcher template + BanglaBERT-BERTScore
   - human-eval sheets (doc 06)
   - read the two competing Bangla Med-VQA papers (doc 02)
5. **Logins:** if towhaelahi/codegiveway OAuth breaks a third time, debug the KaggleFleet token refresh. Their credential files did not refresh after the re-login on 2026-11-21.

## Important fixes made this session (don't regress)
| Fix | Where |
|---|---|
| transformers 5 re-tied BanglaT5's `lm_head` to the input embeddings during `resize_token_embeddings` (loss 1342 instead of 12, nothing learns). `load_model` now keeps untied checkpoints untied and raises if tied. | `medthink_bn.py` `load_model`, doc 08 |
| v2 protocol: val split (seed 0), epoch caps 40/60/12, patience 5/5/3, **`min_epochs` = 25% of cap** (10/15/3), best-weight restore | `medthink_bn.py` |
| Segment loss (`ans_weight`), choice scoring, image ablation, gate bias, vision warm-up | `medthink_bn.py`, doc 08 |
| Kaggle CLI exits 0 on rejected pushes ("Maximum batch GPU session count of 2 reached") → hub now reads stdout | `kaggle_hub/kaggle_runner.py` `push_job` |
| A failed status query (network/404) no longer marks a live run ERROR (it nearly caused a re-push over a running kernel) | `kaggle_runner.py` `check_status` |
| OAuth accounts isolated via a per-account `HOME` (kagglesdk ignores `KAGGLE_CONFIG_DIR`); `kernels push -t` (run-time cap) only when asked; no double push | `kaggle_hub/account_manager.py`, `kaggle_runner.py` |
| v2b protocol: patience 15/15/5, `min_epochs` 50% (doc 10); slug prefix `mtbn-v2b-`; `fetch_results.py` downloads results only | `medthink_bn.py`, `experiments/` |
| Web-panel sign-in: the old 8080 server predated the `/api/accounts/login` route (404). Added **Sign in with a link** (start/finish/cancel; open the link in a private window to choose another Kaggle account, then paste the code). Restart the panel after editing `kaggle_hub/` | `account_manager.py` `start/finish/cancel_link_login`, `server/api_handlers.py`, `static/`, `tests/test_link_login.py` |
| Web panel rebuilt (no Tailwind), tokens no longer sent to the browser, GPU-hour estimates per account | `kaggle_hub/server/` |

Tests: `python3 -m unittest discover kaggle_hub/tests` (70 pass). Library checks: scratch scripts only (`test_lib.py`, `test_train.py`, `verify_fix.py`, which need torch/transformers in a venv); not committed.

## File map
- `medthink_bn.py`: library (CRLF line endings; keep them). `build_notebooks.py` regenerates `notebooks/*.ipynb` from it, so always re-run after editing the library.
- `experiments/launch_runs.py`: per-run notebook generation + push + `--schedule`. Generated notebooks: `experiments/generated/`.
- `../kaggle-fleet/` (**KaggleFleet**, package `kaggle_fleet`, state `~/.kaggle_fleet`): the multi-account tool, `kfleet account list|test|login`, `kfleet serve`. It replaces the in-project `kaggle_hub/`, which is deprecated and kept for reference.
- `backup/`: pre-v2 copies of the library, builder and notebooks.
- Old (paper-protocol) results: `all_results.csv`, `metrics_json/`, `paper_tables/`; they stay as the replication appendix.
