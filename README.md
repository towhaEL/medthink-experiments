# MedThink on Bengali Med-VQA

Evaluating **MedThink** (Gai et al., *Findings of NAACL 2025*) on Bengali
Med-VQA data: `R-RAD-Bengali`, `R-SLAKE-Bangla`, `R-PathVQA-Bengali`.

Reference implementation: <https://github.com/Tang-xiaoxiao/Medthink>

---

## Files

| File | Purpose |
|---|---|
| `medthink_bn.py` | The whole implementation: data unification, DETR fusion model, training engine, Bengali metrics |
| `build_notebooks.py` | Regenerates the notebooks from the library. Edit the library, re-run this, re-upload |
| `notebooks/00_data_prep.ipynb` | Validation + manifests (CPU) |
| `notebooks/01_extract_detr_features.ipynb` | DETR `(100, 256)` feature cache (GPU, once) |
| `notebooks/02_train_closed_end.ipynb` | Accuracy — paper Table 2 |
| `notebooks/03_train_open_end.ipynb` | BLEU/ROUGE — paper Table 3 |
| `notebooks/04_two_stage_reasoning.ipynb` | The two-stage strategy |
| `notebooks/05_results_report.ipynb` | Aggregated tables (CPU) |

The library is embedded into every notebook via `%%writefile`, so each notebook
runs standalone with no extra dataset attachment.

---

## Design decisions

### The backbone had to change

MedThink's TextualEncoder/Decoder is **UnifiedQA (T5-base)**. Its SentencePiece
vocabulary is English-only, so Bengali input degenerates to `<unk>` and the
model learns nothing. We substitute **`csebuetnlp/banglat5`** (247M), which is
Bengali-native and the closest match to the paper's 223M budget.

Notebook 00 §6 measures this directly (tokens/word and unk-rate for both
tokenizers). That table is the justification for the substitution and belongs
in your paper.

Because BanglaT5 is 247M against the paper's 223M, its **"0.2B parameters"
comparison survives** — worth stating explicitly in your paper. Switching to
`google/mt5-base` (580M, mostly the 250k multilingual embedding table) breaks
that claim; report your own count if you do.

**BanglaT5 is T5 v1.1** (`feed_forward_proj: "gated-gelu"`,
`tie_word_embeddings: false`), a markedly less stable family than the T5 v1.0
that UnifiedQA is built on — as is mT5. See the learning rate note below.

### Learning rate: 1e-4, not the paper's 5e-4

The paper's 5e-4 was tuned for T5 v1.0. On a v1.1 backbone it drives training
to NaN within a few optimiser steps — which is exactly what happened on the
first run here. 1e-4 is the standard fine-tuning rate for the v1.1 family.
**This is a documented deviation** — state it in your paper.

### The visual gate starts closed

With a naive initialisation the gate sits at `sigmoid(0) = 0.5`, so half the
pretrained encoder output is replaced by a randomly-projected image signal at
step 0. That wrecks the pretrained representation before the model has learned
anything and, combined with a high learning rate, diverges to NaN.

The gate bias therefore starts at -3 (`lambda ~= 0.047`): the model begins as
essentially pure text BanglaT5 and *learns* to open the gate to vision. Notebook
02/03/04 print `mean gate lambda` in the pre-flight check; it should start
near 0.05.

### Switching backbones

`model_name` in the config cell is the only change:

```python
model_name = "csebuetnlp/banglat5"  # default: 247M, Bengali-only
model_name = "google/mt5-base"      # 580M, multilingual -> also set bs=2, ga=16
```

**Notebooks 00 and 01 do not need re-running** — manifests and DETR features
are independent of the text backbone. Only 02/03/04 re-run. Keep `lr = 1e-4`
either way: both backbones are T5 v1.1.

Results files are keyed by backbone (`rad_closed-end_Explanation_banglat5`), so
runs from both models coexist and notebook 05 tabulates them side by side.

### Three schema traps in your data

Handled in `unify_records`; they corrupt results silently if missed.

1. **R-PathVQA open-end stores `answer` as a raw string**, and its `choices` is
   a vestigial `["হ্যাঁ","না"]` left from the closed-end template. R-RAD and
   R-SLAKE open-end store an **integer index** into a one-element `choices`.
   Reading them the same way makes "হ্যাঁ" the gold answer for every PathVQA
   open-end question.
2. **R-SLAKE keys images as `img_name`** (`xmlab1/source.jpg`); the other two
   use `image`.
3. One R-RAD open-end train record has an empty `solution`; one R-PathVQA
   closed record has English `["yes","no"]`.

### Metrics

English BLEU/ROUGE tooling mangles Bengali: whitespace splitting leaves the
danda (`।`) glued to words, and `rouge_score` applies a Porter stemmer against
a Latin alphabet. All metrics here are implemented from scratch over a Bengali
Unicode-block tokenizer, with Bengali digits normalised to ASCII and
nukta/ZWNJ variants unified.

Open-end results are reported at **two granularities**:

* `full/*` — whole generation vs whole target (answer + rationale)
* `ans/*` — parsed answer vs gold answer

The paper's Table 3 must be the *full* variant: R-SLAKE open-end answers
average 1.8 tokens, which admit no 4-grams at all, so its reported BLEU-4 of
10.3 cannot come from answer-only scoring. The `ans/*` numbers are the
clinically meaningful ones — a model can score well on rationale overlap while
getting the diagnosis wrong.

### Dynamic padding — a ~3-4x speedup for free

The reference repo pads every example to `source_len=512`. Measured on this
data with BanglaT5's tokenizer, real inputs are **11-78 tokens** (p50 ~20), so
89% of every encoder batch was padding — 512 was inherited from long-context
ScienceQA, not short medical questions.

Batches are now padded to their own longest member. Attention masks and the
-100 label mask make this exactly equivalent to fixed padding. Mean padded
width per batch of 8:

| dataset/subset | source | target | encoder positions | cross-attn work |
|---|---|---|---|---|
| rad/closed-end | 29 | 157 | 17.5x fewer | 35.8x less |
| rad/open-end | 17 | 187 | 29.5x | 50.6x |
| slake/closed-end | 28 | 146 | 18.0x | 39.6x |
| slake/open-end | 16 | 161 | 32.6x | 64.7x |
| path/closed-end | 43 | 194 | 12.0x | 19.7x |
| path/open-end | 20 | 194 | 25.5x | 42.1x |

Wall-clock gain is smaller than those ratios — decoder self-attention still
dominates and its width only falls from 320 to ~170 — so expect **roughly
3-4x**, not 20x. `source_len` and `target_len` are now truncation caps only;
nothing in the data actually reaches them.

`batch_size` is correspondingly 8 x 4 (still effective batch 32).

### `target_len` = 320, not 64

The reference repo's `target_len=64` was sized for English answers. Your
Bengali rationales average 67–81 words (p95 up to 123), so 64 would truncate
every one. Notebook 00 §5 prints the distribution; raise it further if your
p95 grows.

---

## Data to attach on Kaggle

**Your Bengali dataset** (`R-RAD-Bengali/`, `R-SLAKE-Bangla/`, `R-PathVQA-Bengali/`).

**Images** — none ship with your JSONs. Best single source:

| Dataset | Covers | Notes |
|---|---|---|
| `towhaelahi/medthink-dataset` | all three | 1.04 GB; original MedThink release *with* images, and it has the val splits your R-PathVQA-Bengali also has |
| `shashankshekhar1205/vqa-rad-visual-question-answering-radiology` | VQA-RAD | `VQA_RAD Image Folder/synpic*.jpg` |
| `nguyenan301/slake-medical-vqa` | SLAKE | `Slake1.0/imgs/xmlab1/source.jpg` — exact key match |
| `reasat/pathvqa` | PathVQA | `pvqa/images/test/test_0000.jpg`, byte-identical to towhaelahi's |

The image resolver indexes by basename, `parent/basename` and
`grandparent/parent/basename`, so any of these layouts resolves. Notebook 00
**fails loudly below 95% coverage** — without it you are silently running a
text-only ablation on zero-filled visual features.

---

## Run order

```
00_data_prep            CPU     ~5 min    → save output as  medthink-bn-manifests
01_extract_detr_features GPU    ~20-40min → save output as  medthink-bn-features
02 / 03 / 04            GPU     per run   → save outputs as medthink-bn-runs
05_results_report       CPU     ~1 min
```

Notebooks 02–04 have one config cell at the top. Change `dataset` and
`strategy`, run, repeat.

**`quick_run = True` is the default.** It subsamples to 200 train / 100 test
and 2 epochs so you can verify the whole pipeline in minutes. Set it to `False`
for real runs — notebook 05 flags any run still carrying the flag.

### Staged plan

You chose to validate on R-RAD first. Recommended order:

1. `02` with `rad` × `without_R` → the baseline every delta is measured against
2. `02` with `rad` × `Explanation` → the paper's best closed-end strategy
3. `02` with `rad` × `Reasoning`, then `04` with `rad`
4. `03` for the open-end R-RAD counterparts
5. Repeat for `slake`, then `path`

Full replication is 3 datasets × 2 subsets × 4 strategies = **24 runs**.

### Compute budget — read this before planning runs

Even with dynamic padding, the paper's epoch counts are expensive on one T4.
Extrapolated from a measured quick run, per training run:

| Run | samples x epochs | before | est. after |
|---|---|---|---|
| R-RAD closed-end | 1823 x 150 | ~19 h | ~5 h |
| R-SLAKE closed-end | 1943 x 300 | ~39 h | ~10 h |
| R-Path closed-end | 9428 x 50 | ~31 h | ~8 h |

At 8 runs per dataset that is still well over 100 GPU-hours for the full 24-run
grid — several weeks at 30 h/week. **Decide the epoch budget deliberately
before committing.** The paper's 150-300 epochs over ~2k examples is far past
where most of the convergence happens; cutting to 30-50 epochs with the loss
curve reported would be a defensible, clearly-stated deviation, and is what
makes the full grid feasible. R-PathVQA ships val splits you can early-stop on;
R-RAD and R-SLAKE do not.

### Kaggle's limits

30 GPU-hours/week, 12h per session. Every run checkpoints each epoch and stops
at `TIME_BUDGET_H = 10.5`; **re-run the same notebook with the same config and
it resumes from the checkpoint**. `train_info.stopped_early` records truncation
and notebook 05 refuses to treat those runs as final.

The paper trained on 4×3090 (R-RAD 2.5h, R-SLAKE 5.5h, R-Path 14h). On a single
P100/T4 expect several times that, which is why R-Path is staged last.

`fp16` stays **off**: T5-family models overflow to NaN in fp16. On a T4, `bf16`
is unavailable; on newer accelerators you may enable bf16 instead.

Default batch is `4 x 8` (effective 32, the paper's setting), which BanglaT5
fits on a 16GB T4. mT5-base in fp32 with AdamW needs ~9GB for weights,
gradients and optimiser state before activations, so switch to `2 x 16` there.

---

## Troubleshooting

**`loss nan` / accuracy 0.00 with `Unresolved 100.00`**
Training diverged; the model emits empty strings. The pre-flight check in §2b
of notebooks 02–04 catches this before you spend hours on it. If it still
happens: lower `cfg.lr` to 5e-5, **delete the checkpoint** (`ckpt_*.pt` — a
NaN checkpoint reloads as NaN), and re-run. `train_info` now records
`nan_batches` and `diverged`, and notebook 05 refuses to treat such runs as
results.

**Empty predictions but finite loss**
Usually `target_len` truncation. Check notebook 00 §5 length stats.

**CUDA OOM**
Halve `batch_size` and double `grad_accum` to keep the effective batch at 32.

---

## Reference numbers (English MedThink)

Closed-end accuracy, paper Table 2:

| Strategy | R-RAD | R-SLAKE | R-Path |
|---|---|---|---|
| w/o rationale | 79.0 | 82.5 | 86.0 |
| Reasoning | 73.9 | 80.8 | 83.1 |
| Two-Stage | 80.5 | 79.1 | 87.2 |
| **Explanation** | **83.5** | **86.3** | 87.0 |

Expect Bengali numbers to land below these: BanglaT5 is pretrained on far less
data than UnifiedQA, and the rationales are machine-translated. **The
scientific claim to test is not absolute accuracy but whether the
rationale-vs-baseline delta survives the language transfer** — i.e. whether
Explanation still beats `without_R`, and whether Reasoning still
underperforms it. That comparison is notebook 05 §4.

---

## Known limitations

* R-PathVQA-Bengali splits differ from the paper (9428/3257 vs 9806/3391) and
  add val splits. Report your own counts; don't cite the paper's.
* Rationale quality is bounded by the Bengali translation of the MDMRs, which
  is upstream of this code and unaudited here.
* `generate_with_image` hands `encoder_outputs` to `generate()` rather than
  patching `prepare_inputs_for_generation` as the original MM-CoT/MedThink code
  does. This is deliberate — that patch is the part most likely to break across
  `transformers` versions — but it is the first thing to check if generation
  misbehaves after a library upgrade.
