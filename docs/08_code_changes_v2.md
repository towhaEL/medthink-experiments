# 08 — Code Changes (v2 protocol + AARL switches)

Implemented 2026-11-20 in `medthink_bn.py` and `build_notebooks.py`. Notebooks were regenerated.
Originals are in `backup/` (`medthink_bn.v1.py`, `build_notebooks.v1.py`, `notebooks.v1/`).

**Backward compatible:** with default settings (`protocol="paper"`, `ans_weight=None`, `gate_bias_init=-3.0`, `vision_warmup_epochs=0`, `seed=42`), training, run names and checkpoints are identical to the original 24-run grid. Notebook 04 (Two-Stage) is unchanged in behaviour.

## What changed

| Change | Where | What it does |
|---|---|---|
| **v2 protocol** | `Config.protocol`, `DATASETS[..]["epochs_v2"/"patience_v2"]` | Epoch caps RAD 40 / SLAKE 60 / Path 12 with early stopping. **Since 2026-11-21 ("v2b"): patience 15 / 15 / 5, `min_epochs` 50% of cap** (was 5 / 5 / 3 and 25%, which undertrained, see doc 10) |
| **Validation split** | `split_train_val`, `cap_records` | 10% of train, fixed seed 0 (same val for every run and seed). SLAKE grouped by image (0 shared images, verified); RAD random because 97% of RAD test images are also in train. Path uses its shipped val split. Val capped at 500 items. |
| **Early stopping + best weights** | `train_model(val_ds=...)` | Validates each epoch, saves `ckpt_<run>_best.pt`, restores best weights at the end. Resume-safe: a finished run is not retrained. `early_stopped` (complete) is kept distinct from `stopped_early` (time budget, incomplete). |
| **Segment-balanced loss** | `Config.ans_weight`, `segment_loss`, `answer_segment_mask` | `α·L_ans + (1−α)·L_rat`, each normalised per example. `None` = paper loss (verified identical to HF's loss). |
| **Answer/rationale loss logging** | `history[i]["ans_loss"/"rat_loss"]` | Logged for every run, including vanilla ones. Use it for the loss-dilution figure. |
| **Choice scoring** | `score_choices` → `metrics["ChoiceAccuracy"]` | Closed-end accuracy by option likelihood; never "unresolved". Answer-first strategies only. |
| **Image ablation** | `MedThinkDataset.set_image_mode`, `image_ablation` | Re-scores the model with zeroed images and with shuffled images (never a record's own image; verified 0%). Reported as `ablation/{real,zero,shuffle}/...`. |
| **Gate init** | `Config.gate_bias_init` | −3.0 (λ≈0.047, original) or e.g. −1.0 (λ≈0.27) |
| **Vision warm-up** | `Config.vision_warmup_epochs` | First N epochs train only the visual pathway (verified: 0 text tensors change, all 8 visual tensors do) |
| **Run names** | `Config.run_name` | Suffixes `_v2_s43`, `_aw0.9`, `_gb-1`, `_vw1`. Seeds and ablations never overwrite each other. |
| **Notebook 05** | builder | Run identity includes protocol/seed/AARL switches; new **"1b · v2 mean ± std over seeds"** table → `v2_summary.csv`. Replication tables use paper-protocol runs only. |

## Fix: transformers 5 breaks BanglaT5's output layer (2026-11-21)

Found by the first real-GPU sanity run. Under transformers 5 (current Kaggle image), BanglaT5's config reports `tie_word_embeddings=True` although the checkpoint has a separate `lm_head`. `resize_token_embeddings()` then re-ties `lm_head` to the input embeddings:

| | lm_head tied | params | loss before training |
|---|---|---|---|
| Broken (transformers 5, old `load_model`) | yes | 226.6M | **1342** |
| Fixed (transformers 5.23 and 4.57) | no | 251.3M | **11.9** |

`load_model` now detects untied checkpoints from the loaded weights, keeps them untied, and raises if they end up tied. Genuinely tied checkpoints (UnifiedQA / T5 v1.0, needed for the English control) stay tied. The September 24-run grid was trained under the older transformers and is not affected: its losses fell to ~0.

## How to run (notebooks 02 / 03)

Edit only the config cell:

```python
PRESET   = "full"          # sanity | baseline | full
DATASET  = "rad"           # rad | slake | path
STRATEGY = "Explanation"
SEED     = 42              # run 42, 43, 44
PROTOCOL = "v2"
ANS_WEIGHT     = None      # AARL: 0.8 / 0.9 / 0.95
GATE_BIAS      = -3.0      # AARL: -1.0
VISION_WARMUP  = 0         # AARL: 1
```

Each new run first prints `[ans_mask] OK …` lines; the notebook refuses to train if they don't match. Each epoch then prints `loss … [ans x | rat y]` and `[val] …`.

### Week-1 run list (E1 baselines, closed-end, notebook 02)

| # | DATASET | PRESET / STRATEGY | SEED | Est. T4 time |
|---|---|---|---|---|
| 1–3 | rad | baseline (without_R) | 42, 43, 44 | ~0.4 h each |
| 4–6 | rad | full / Explanation | 42, 43, 44 | ~1.3 h each |
| 7–9 | slake | baseline | 42, 43, 44 | ~0.7 h each |
| 10–12 | slake | full / Explanation | 42, 43, 44 | ~1.8 h each |
| 13 | rad, slake | full / Reasoning | 42 | ~1.5 h each |

Week-2 E2 (α sweep): same as rows 4 and 10 with `ANS_WEIGHT` ∈ {0.5, 0.8, 0.95}, seed 42.

Tip: Kaggle "Save Version → Run all" runs in the background. You can queue several versions with different config cells, up to the account's GPU concurrency limit.

## Verification done (locally, CPU)

Done with a tiny randomly initialised T5, the real BanglaT5 tokenizer and your real data:
- Answer mask exact on **8,000 real items** (5 subsets × 4 strategies × 400): 0 mismatches.
- `segment_loss(alpha=None)` equals HF loss (24.080217 vs 24.080198).
- Early stopping, best-weight restore, finished-run resume, warm-up freezing, choice scoring, ablation, Reasoning val and open-end val all behave as specified.
- Notebooks 02, 03 and 05 executed cell by cell end to end. Notebook 05 reproduces the existing paper-protocol table exactly.

**Not yet verified:** a real BanglaT5 + GPU run. Run one `PRESET="sanity"` with `PROTOCOL="v2"` on Kaggle first (~5 min) before queueing the full list.

## Additions 2026-11-21/22: BiomedCLIP features (E3) and English control (E5)

With the defaults (`FEATURES="detr"`, `LANG="bn"`) every notebook behaves and names runs exactly as before. Both changes were diffed against the previous notebook.

| Change | Where | Notes |
|---|---|---|
| **Notebook 01b**: BiomedCLIP ViT-B/16 (`open_clip`, `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`), `trunk.forward_features` → (197, 768) fp16 per image | `build_notebooks.py` → `notebooks/01b_extract_biomedclip_features.ipynb` | Output `features_biomedclip.npy` + `name_to_row_biomedclip.json`, Kaggle dataset `towhaelahi/medthink-bn-biomedclip` (5131 images, 1.55 GB) |
| `FEATURES = "detr" \| "biomedclip"` config line; `FEATURE_SETS` table; `load_features()` picks the file and asserts the shape | nb 02/03 config + loader | Sets `patch_num/patch_dim` |
| Run name gets `_f{num}x{dim}` for non-DETR features | `Config.run_name` | e.g. `..._aw0.5_f197x768` |
| `LANG = "bn" \| "en"`, `MODEL`, `LR` config lines | nb 02/03 config | E5: `LANG="en"`, `MODEL="allenai/unifiedqa-t5-base"`, attach `towhaelahi/medthink-dataset` |
| `LANG_MARKERS`, `set_language(lang)`: switches `MARK_*`, `OPTION_LETTERS` and the answer/rationale regexes (en: `Question:`/`Options:`/`Answer:`/`Because:`, letters A–F) | `medthink_bn.py` | `MedThinkDataset` asserts `cfg.lang == CURRENT_LANG` |
| `overlay_language(records, ...)`: replaces the Bengali text in manifest records with the parallel English text from `Medthink-Dataset/` | `medthink_bn.py`, called in `BUILD_DS` | Same qids, images, order and splits, so the val/test items are identical. Raises on any id/answer-index/image mismatch. Checked locally on every closed-end split of all 3 datasets: 0 mismatches |
| `Config.lang`; run name gets `_en` | `medthink_bn.py` | e.g. `rad_closed-end_Explanation_unifiedqa-t5-base_v2_s42_en` |
| Launcher: `run(..., features="biomedclip")` → slug `-bmc` + feature dataset; `run(..., lang="en")` → slug `-en`, sets `LANG`/`MODEL`, attaches the English dataset; optional `lr=` → slug `-lr5e4` | `experiments/launch_runs.py` | Plans `bmcsanity`, `e3b`, `ensanity`, `e5`, `pathsanity`, `pathe1` |
| Gate summary printed "init 0.047" even for gate bias −1; now prints the real init | nb 02/03 report cell | Cosmetic |
| `experiments/analyze_v2b.py`: 3-seed table, exact paired McNemar, α sweep, ablation/gate table (stdlib) | new | Produced doc 13 |

Pre-change copies: `backup/pre_bmc/` and `backup/pre_e5/`.

UnifiedQA (T5 v1.0) has genuinely tied embeddings. `load_model`'s transformers-5 guard only enforces untying for checkpoints that load untied, so UnifiedQA passes through unchanged.

## Not implemented yet (next steps)

- ~~BiomedCLIP feature extraction~~ done (see above)
- ~~Same-pipeline English control~~ done (see above); sanity run pending
- Rationale-length ablation (first 1–2 sentences)
- Zero-shot VLM baseline notebook (Qwen2.5-VL-3B / MedGemma-4B)

## 2026-11-24: rationale dropout (`rat_drop`)
- **What:** `Config.rat_drop` (default 0.0). During training, each draw of an Explanation item gets an answer-only target (`without_R` target and answer mask) with this probability, resampled every epoch. The input text is unchanged. Validation/test datasets never drop (`MedThinkDataset.rat_drop` is set by `train_model` and reset before it returns).
- **Why:** the obvious simple alternative to the segment-balanced loss (doc 04 lists it as a required baseline). It uses the vanilla token-mean loss, so it tests "fewer rationale tokens" against "per-segment normalisation".
- **Naming:** run name `_rd0.5`; slug `-rd50`; notebook switch `RAT_DROP`; the checkpoint resume check includes `rat_drop`. Default runs keep their names.
- **Where:** `medthink_bn.py` (Config, run_name, MedThinkDataset.__getitem__, train_model), `build_notebooks.py` (both training templates + results table), `experiments/launch_runs.py` (slug, render, plans `rdsanity` / `rd`). Pre-change copies: `backup/pre_ratdrop/`.
- **Checked locally** (stub tokenizer): ~49% of draws are answer-only, the answer-only target equals the `without_R` target, inputs are identical, eval and `without_R` are unaffected.
