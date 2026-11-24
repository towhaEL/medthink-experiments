"""
build_notebooks.py
==================
Generates the Kaggle notebooks for the Bengali MedThink study.

`medthink_bn.py` is the single source of truth; it is embedded into every
notebook via `%%writefile` so each notebook runs standalone on Kaggle with no
extra dataset attachment. Edit the library, re-run this script, re-upload.

    python build_notebooks.py
"""

import io
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "notebooks"
OUT.mkdir(exist_ok=True)

LIB = (HERE / "medthink_bn.py").read_text(encoding="utf-8")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def notebook(cells, accelerator="GPU"):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.13"},
            "accelerator": accelerator,
            "kaggle": {"accelerator": "nvidiaTeslaT4" if accelerator == "GPU" else "none",
                       "dataSources": [], "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, nb: dict):
    path = OUT / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {path}  ({len(nb['cells'])} cells)")


LIB_CELL = code("%%writefile medthink_bn.py\n" + LIB)

SETUP = """
import os, sys, json, glob, math, time
from pathlib import Path
import numpy as np

# Optional: BanglaT5's official normaliser. Falls back gracefully if offline.
try:
    import normalizer  # noqa: F401
except ImportError:
    !pip install -q git+https://github.com/csebuetnlp/normalizer.git 2>/dev/null || echo "normalizer unavailable - using NFKC fallback"

import importlib, medthink_bn
importlib.reload(medthink_bn)
import medthink_bn as M

print("csebuetnlp normalizer:", M._HAS_CSEBUETNLP)
DATA_ROOT = "/kaggle/input"
WORK = "/kaggle/working"
"""

# --------------------------------------------------------------------------- #
# 00 - data preparation
# --------------------------------------------------------------------------- #

nb00 = notebook([
    md("""
# 00 · Data Preparation & Validation
### MedThink (Findings of NAACL 2025) on Bengali Med-VQA

Builds the unified manifests every later notebook consumes, and validates the
three Bengali releases against the paper's Table 1.

**Attach as data sources**
1. Your Bengali dataset (`R-RAD-Bengali`, `R-SLAKE-Bangla`, `R-PathVQA-Bengali`)
2. Images — `towhaelahi/medthink-dataset` covers all three. Fallbacks:
   `shashankshekhar1205/vqa-rad-visual-question-answering-radiology`,
   `nguyenan301/slake-medical-vqa`, `reasat/pathvqa`

**Outputs** → `manifests/*.json`, `unique_images.json`. Save this notebook's
output as a Kaggle dataset named **`medthink-bn-manifests`**.

No GPU needed — run this on CPU to save quota.
"""),
    LIB_CELL,
    code(SETUP + """
print("\\nAttached inputs:")
for p in sorted(Path(DATA_ROOT).glob("*")):
    print("   ", p.name)
"""),
    md("## 1 · Locate the Bengali dataset directories"),
    code("""
found = {}
for ds, meta in M.DATASETS.items():
    hits = sorted(Path(DATA_ROOT).glob(f"**/{meta['dir']}"))
    found[ds] = str(hits[0]) if hits else None
    print(f"{ds:6s} {meta['dir']:22s} -> {found[ds]}")

missing = [d for d, v in found.items() if v is None]
assert not missing, (
    f"Missing Bengali dataset directories for: {missing}. "
    "Attach the dataset containing them before continuing."
)
"""),
    md("""
## 2 · Index every attached image

Filenames differ between image sources, so each file is indexed under several
keys (`synpic54610.jpg`, `xmlab1/source.jpg`, `Train_images/train_0157.jpg`).
Whichever image dataset you attach, lookups resolve.
"""),
    code("""
img_index = M.build_image_index([DATA_ROOT])

# Show what the three reference styles resolve to.
for probe in ["synpic54610.jpg", "xmlab1/source.jpg", "train_0157.jpg"]:
    print(f"  {probe:24s} -> {M.resolve_image(probe, img_index)}")
"""),
    md("## 3 · Unify the three schemas and resolve images"),
    code("""
manifests, rows = {}, []

for ds in M.DATASETS:
    for sub in ["closed-end", "open-end"]:
        for split in M.DATASETS[ds]["splits"][sub]:
            raw = M.load_split(DATA_ROOT, ds, sub, split)
            recs = M.unify_records(raw, ds, sub, split)

            hit = 0
            for r in recs:
                p = M.resolve_image(r["image_key"], img_index)
                r["image_path"] = p or ""
                hit += bool(p)

            manifests[f"{ds}_{sub}_{split}"] = recs
            rows.append({
                "dataset": ds, "subset": sub, "split": split,
                "n": len(recs),
                "images": len({r["image_key"] for r in recs}),
                "img_found_%": round(100 * hit / max(len(recs), 1), 1),
            })

hdr = f"{'dataset':8s}{'subset':12s}{'split':7s}{'n':>7s}{'images':>8s}{'img_found_%':>13s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['dataset']:8s}{r['subset']:12s}{r['split']:7s}"
          f"{r['n']:>7d}{r['images']:>8d}{r['img_found_%']:>13.1f}")
"""),
    md("""
### Image coverage gate

Zero-coverage means the visual pathway is fed zeros and you are silently
running a text-only ablation. This cell fails loudly rather than letting that
happen unnoticed.
"""),
    code("""
bad = [r for r in rows if r["img_found_%"] < 95.0]
if bad:
    print("!! Low image coverage — attach the missing image dataset:\\n")
    for r in bad:
        print(f"   {r['dataset']}/{r['subset']}/{r['split']}: {r['img_found_%']}%")
    print("\\n   towhaelahi/medthink-dataset            (all three)")
    print("   shashankshekhar1205/vqa-rad-...        (VQA-RAD)")
    print("   nguyenan301/slake-medical-vqa          (SLAKE)")
    print("   reasat/pathvqa                         (PathVQA)")
else:
    print("All splits resolve >=95% of their images.")
"""),
    md("## 4 · Compare against the paper's Table 1"),
    code("""
# Paper Table 1: (train, test) per dataset/subset.
PAPER = {
    ("rad", "closed-end"):   (1823, 272),
    ("rad", "open-end"):     (1241, 179),
    ("slake", "closed-end"): (1943, 416),
    ("slake", "open-end"):   (2976, 645),
    ("path", "closed-end"):  (9806, 3391),
    ("path", "open-end"):    (9933, 3364),
}

print(f"{'dataset/subset':22s}{'ours tr':>9s}{'paper tr':>10s}{'ours te':>9s}{'paper te':>10s}  match")
print("-" * 72)
for (ds, sub), (ptr, pte) in PAPER.items():
    otr = len(manifests.get(f"{ds}_{sub}_train", []))
    ote = len(manifests.get(f"{ds}_{sub}_test", []))
    ok = "yes" if (otr, ote) == (ptr, pte) else "differs"
    print(f"{ds+'/'+sub:22s}{otr:>9d}{ptr:>10d}{ote:>9d}{pte:>10d}  {ok}")

print("\\nNote: R-PathVQA-Bengali also ships val splits, which the paper's table")
print("does not list; its train/test counts differ from the paper accordingly.")
"""),
    md("## 5 · Length statistics — these set `source_len` and `target_len`"),
    code("""
print(f"{'dataset/subset':22s}{'q_tok':>7s}{'ans_tok':>9s}{'sol_tok':>9s}{'sol_p95':>9s}{'sol_max':>9s}")
print("-" * 66)
for ds in M.DATASETS:
    for sub in ["closed-end", "open-end"]:
        recs = manifests[f"{ds}_{sub}_train"]
        q = [len(M.bn_tokenize(r["question"])) for r in recs]
        a = [len(M.bn_tokenize(r["answer"])) for r in recs]
        s = sorted(len(M.bn_tokenize(r["solution"])) for r in recs)
        print(f"{ds+'/'+sub:22s}{np.mean(q):>7.1f}{np.mean(a):>9.1f}"
              f"{np.mean(s):>9.1f}{s[int(.95*len(s))]:>9d}{s[-1]:>9d}")

print("\\nRationales run to ~75 words. The reference repo's target_len=64 was sized")
print("for English answers only and would truncate every Bengali rationale, so the")
print("Config default is 320.")
"""),
    md("""
## 6 · Why the backbone had to change

The paper's TextualEncoder/Decoder is UnifiedQA (T5-base). Its SentencePiece
vocabulary is English-only. This cell measures what that costs on Bengali —
the evidence for substituting BanglaT5, and a table worth putting in your paper.
"""),
    code("""
from transformers import AutoTokenizer

sample = [r["question"] for r in manifests["rad_closed-end_train"][:300]]
sample += [r["solution"] for r in manifests["rad_closed-end_train"][:300]]

print(f"{'tokenizer':34s}{'tok/word':>10s}{'unk %':>9s}")
print("-" * 53)
for name in ["t5-base", "google/mt5-base", "csebuetnlp/banglat5"]:
    try:
        tk = AutoTokenizer.from_pretrained(name, use_fast=False)
    except Exception as e:
        print(f"{name:34s}  unavailable ({type(e).__name__})")
        continue
    n_tok = n_word = n_unk = 0
    for s in sample:
        ids = tk(s, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        n_word += len(s.split())
        n_unk += sum(1 for i in ids if i == tk.unk_token_id)
    print(f"{name:34s}{n_tok/max(n_word,1):>10.2f}{100*n_unk/max(n_tok,1):>9.2f}")

print("\\nA high unk rate or an extreme token/word ratio means the tokenizer cannot")
print("represent Bengali: the model would train on <unk> soup. This is the")
print("justification for the one substitution we make to the paper's architecture.")
"""),
    md("## 7 · Save manifests"),
    code("""
mdir = Path(WORK) / "manifests"
mdir.mkdir(exist_ok=True, parents=True)

for key, recs in manifests.items():
    with open(mdir / f"{key}.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False)

# Unique images, for notebook 01 to featurise exactly once.
uniq = {}
for recs in manifests.values():
    for r in recs:
        if r["image_path"]:
            uniq[r["image_key"].lower()] = r["image_path"]
with open(Path(WORK) / "unique_images.json", "w", encoding="utf-8") as fh:
    json.dump(uniq, fh, ensure_ascii=False, indent=1)

with open(Path(WORK) / "data_summary.json", "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=1)

print(f"{len(manifests)} manifests written to {mdir}")
print(f"{len(uniq)} unique images to featurise")
"""),
    md("""
---
**Next:** save this notebook's output as a dataset called **`medthink-bn-manifests`**
(Output → *New Dataset*), then run `01_extract_detr_features`.
"""),
], accelerator="None")

write("00_data_prep.ipynb", nb00)


# --------------------------------------------------------------------------- #
# 01 - DETR feature extraction
# --------------------------------------------------------------------------- #

nb01 = notebook([
    md("""
# 01 · DETR Visual Features
### `VisualEncoder(·)` from MedThink Sec. 3.2

Extracts DETR encoder-decoder features for every unique image, shape
**(100, 256)** — 100 object queries × 256 hidden dims. This is exactly the
`(100, 256)` hardcoded in the reference implementation's
`T5ForMultimodalGeneration.from_pretrained(..., (100, 256))`.

Runs **once**; all 24 training runs reuse the cache.

**Attach:** `medthink-bn-manifests` (from notebook 00) + your image datasets.
**Outputs:** `features.npy`, `name_to_row.json` → save as **`medthink-bn-features`**.
"""),
    LIB_CELL,
    code(SETUP + """
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""),
    md("## 1 · Load the image list from notebook 00"),
    code("""
cands = sorted(Path(DATA_ROOT).glob("**/unique_images.json"))
assert cands, "unique_images.json not found — attach the medthink-bn-manifests dataset."
uniq = json.load(open(cands[0], encoding="utf-8"))
print(f"{len(uniq)} unique images from {cands[0]}")

# Paths were recorded in notebook 00's session; re-resolve in case the mount
# points moved between sessions.
img_index = M.build_image_index([DATA_ROOT])
keys, paths, missing = [], [], []
for k in uniq:
    p = uniq[k] if os.path.exists(uniq[k]) else M.resolve_image(k, img_index)
    (keys.append(k) or paths.append(p)) if p else missing.append(k)

print(f"resolved {len(keys)} | missing {len(missing)}")
if missing:
    print("  e.g.", missing[:5])
"""),
    md("## 2 · Load DETR"),
    code("""
from transformers import DetrModel, DetrImageProcessor

DETR_NAME = "facebook/detr-resnet-101"      # paper cites Carion et al. 2020
processor = DetrImageProcessor.from_pretrained(DETR_NAME)
detr = DetrModel.from_pretrained(DETR_NAME).to(DEVICE).eval()

print("DETR params: %.1fM" % (sum(p.numel() for p in detr.parameters()) / 1e6))
print("d_model:", detr.config.d_model, "| queries:", detr.config.num_queries)
assert (detr.config.num_queries, detr.config.d_model) == (100, 256), \\
    "Expected (100, 256) to match the reference implementation."
"""),
    md("""
## 3 · Extract

`last_hidden_state` of the DETR **decoder** is `(B, 100, 256)` — one vector per
object query. Stored as float16: the values are small, and halving the cache
keeps it comfortably inside a Kaggle dataset.
"""),
    code("""
BATCH = 8
feats = np.zeros((len(keys), 100, 256), dtype=np.float16)
t0 = time.time()
failed = []

for i in range(0, len(keys), BATCH):
    chunk_paths = paths[i:i + BATCH]
    images = []
    for p in chunk_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            failed.append(p)
            images.append(Image.new("RGB", (224, 224)))   # zero-ish placeholder

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = detr(**inputs)
    feats[i:i + len(images)] = out.last_hidden_state.cpu().numpy().astype(np.float16)

    if (i // BATCH) % 25 == 0:
        done = i + len(images)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done}/{len(keys)}  {rate:.1f} img/s  eta {(len(keys)-done)/max(rate,1e-9)/60:.1f} min")

print(f"\\ndone in {(time.time()-t0)/60:.1f} min | unreadable: {len(failed)}")
"""),
    md("## 4 · Verify and save"),
    code("""
print("shape:", feats.shape, "dtype:", feats.dtype)
print("mean %.4f  std %.4f  min %.3f  max %.3f"
      % (feats.mean(), feats.std(), feats.min(), feats.max()))

# All-zero rows would mean an image silently failed to encode.
zero_rows = int((np.abs(feats).sum(axis=(1, 2)) == 0).sum())
print("all-zero feature rows:", zero_rows)
assert zero_rows <= len(failed), "Unexpected zero features — check image loading."

np.save(Path(WORK) / "features.npy", feats)
with open(Path(WORK) / "name_to_row.json", "w", encoding="utf-8") as fh:
    json.dump({k: i for i, k in enumerate(keys)}, fh, ensure_ascii=False)

mb = (Path(WORK) / "features.npy").stat().st_size / 1e6
print(f"saved features.npy ({mb:.0f} MB) + name_to_row.json")
"""),
    md("""
---
**Next:** save this output as **`medthink-bn-features`**, then run
`02_train_closed_end`.
"""),
])

write("01_extract_detr_features.ipynb", nb01)



# --------------------------------------------------------------------------- #
# Shared snippets for the training notebooks
# --------------------------------------------------------------------------- #

LOADER = """
def load_manifest(ds, sub, split):
    c = sorted(Path(DATA_ROOT).glob(f"**/manifests/{ds}_{sub}_{split}.json"))
    assert c, f"manifest {ds}_{sub}_{split}.json not found - attach medthink-bn-manifests"
    return json.load(open(c[0], encoding="utf-8"))

def load_features():
    f = sorted(Path(DATA_ROOT).glob("**/features.npy"))
    n = sorted(Path(DATA_ROOT).glob("**/name_to_row.json"))
    assert f and n, "features not found - attach medthink-bn-features"
    return np.load(f[0], mmap_mode="r"), json.load(open(n[0], encoding="utf-8"))

def subsample(recs, n, seed=0):
    import random as _r
    if n is None or n >= len(recs):
        return recs
    g = _r.Random(seed)
    return g.sample(recs, n)
"""

BUILD_DS = """
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
features, name_to_row = load_features()

train_recs = load_manifest(cfg.dataset, cfg.subset, "train")
test_recs  = load_manifest(cfg.dataset, cfg.subset, "test")

if cfg.quick_run:
    train_recs = subsample(train_recs, 200, cfg.seed)
    test_recs  = subsample(test_recs, 100, cfg.seed)
    print("QUICK RUN: subsampled to", len(train_recs), "train /", len(test_recs), "test")

train_ds = M.MedThinkDataset(train_recs, tokenizer, cfg, features, name_to_row)
test_ds  = M.MedThinkDataset(test_recs,  tokenizer, cfg, features, name_to_row)

print(f"train {len(train_ds)} | test {len(test_ds)}")
_ = train_ds[0]
print(f"missing image features in first sample batch: {train_ds.missing}")
print("\\n--- example input ---\\n" + M.build_input_text(train_recs[0], cfg.strategy))
print("\\n--- example target ---\\n" + M.build_target_text(train_recs[0], cfg.strategy)[:300])
"""

MODEL_CELL = """
M.set_seed(cfg.seed)
model = M.load_model(cfg, tokenizer)
p = M.count_parameters(model)
print(f"total {p['total_M']:.1f}M | trainable {p['trainable_M']:.1f}M "
      f"| visual head {p['visual_head_M']:.1f}M")
print()
print("BanglaT5 is ~247M against the paper's 223M, so its '0.2B parameters'")
print("comparison still holds. (mT5-base would be ~580M, mostly the 250k")
print("multilingual embedding table -- report your own count if you switch.)")
"""


# --------------------------------------------------------------------------- #
# 02 - closed-end
# --------------------------------------------------------------------------- #

nb02 = notebook([
    md("""
# 02 · Closed-End Training — Accuracy (paper Table 2)

Fine-tunes BanglaT5 + DETR gated fusion on closed-end questions and reports
accuracy, for the strategies `without_R`, `Explanation` and `Reasoning`.
Two-Stage Reasoning needs its own pipeline — see notebook 04.

**Attach:** `medthink-bn-manifests`, `medthink-bn-features`.
**Run once per (dataset × strategy).** Change the config cell, run, repeat.

Checkpoints save every epoch, so if Kaggle cuts the session at 12h just re-run
this notebook with the same config and it resumes.
"""),
    LIB_CELL,
    code(SETUP + """
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
print("cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""" + LOADER),
    md("## ⚙️ Config — the only cell you edit between runs"),
    code("""
# ===========================================================================
# CHANGE THESE THREE LINES ONLY.
# ===========================================================================
PRESET   = "sanity"        # sanity | baseline | full
DATASET  = "rad"           # rad | slake | path
STRATEGY = "Explanation"   # without_R | Explanation | Reasoning  (used by "full")
# ===========================================================================
#
#   sanity   200 examples x 60 epochs, without_R      ~5 min
#            375 optimiser steps -- 30x the default smoke test. Proves the
#            pipeline can actually learn. NOT a result; never report it.
#
#   baseline full data, without_R, paper epochs       ~1-2 h
#            The no-rationale number every delta in notebook 05 is measured
#            against. Run this before any rationale strategy.
#
#   full     full data, STRATEGY above, paper epochs  ~6 h
#            The real experiment.
#

_PRESETS = {
    "sanity":   dict(strategy="without_R", epochs=60,   quick_run=True),
    "baseline": dict(strategy="without_R", epochs=None, quick_run=False),
    "full":     dict(strategy=STRATEGY,    epochs=None, quick_run=False),
}
assert PRESET in _PRESETS, f"PRESET must be one of {list(_PRESETS)}"

cfg = M.Config(
    dataset    = DATASET,
    subset     = "closed-end",
    **_PRESETS[PRESET],

    # Swap backbone here. BanglaT5 is 247M (closest to the paper's 223M) and
    # Bengali-only; mT5-base is 580M and multilingual. Both are T5 v1.1, so
    # both need the low lr below. Notebooks 00 and 01 do NOT need re-running.
    model_name = "csebuetnlp/banglat5",  # or "google/mt5-base" (then bs=4, ga=8)

    batch_size = 8,               # dynamic padding shrank sequences ~17x
    grad_accum = 4,               # 8 x 4 = effective batch 32, the paper's setting
    lr         = 1e-4,            # NOT the paper's 5e-4 -- see note below
    source_len = 512,
    target_len = 320,
    num_beams  = 1,
    seed       = 42,
)

TIME_BUDGET_H = 10.5              # stop and checkpoint before Kaggle's 12h cap

# quick_run defaults to 2 epochs only when no number was given.
if cfg.quick_run and cfg.epochs is None:
    cfg.epochs = 2

assert cfg.subset == "closed-end"
assert cfg.strategy in ["without_R", "Explanation", "Reasoning"], \\
    "Two-Stage Reasoning is notebook 04."

n_train = 200 if cfg.quick_run else {"rad": 1823, "slake": 1943, "path": 9428}[cfg.dataset]
steps = n_train * cfg.resolved_epochs() // (cfg.batch_size * cfg.grad_accum)
print(f"PRESET={PRESET}  ->  {cfg.run_name()}")
print(f"  {n_train} examples x {cfg.resolved_epochs()} epochs = ~{steps} optimiser steps")
if PRESET == "sanity":
    print("  SANITY CHECK -- diagnostic only, do not report these numbers.")
print()
print(cfg.to_json())
CKPT = f"{WORK}/ckpt_{cfg.run_name()}.pt"

# /kaggle/working is wiped between sessions. If the previous session's
# output is attached, pull its checkpoint in so a long run resumes
# instead of silently restarting from epoch 0.
RESUME = M.find_checkpoint(CKPT, DATA_ROOT)
"""),
    md("## 1 · Data"),
    code(BUILD_DS),
    md("## 2 · Model — BanglaT5 + cross-attention + gated fusion"),
    code(MODEL_CELL),
    md("""
## 2b · Pre-flight check — is the forward pass finite?

A NaN loss produces a model that generates empty strings and scores 0.00 on
everything, which is easy to mistake for a bad result rather than a broken run.
This traces one forward pass and shows exactly where any non-finite value first
appears, before you spend hours training.

`mean gate lambda` should start near **0.05**. The visual pathway is
deliberately initialised almost closed so the randomly-initialised image
projection cannot overwrite the pretrained encoder at step 0 — that, combined
with the paper's lr of 5e-4, is what drives a T5 v1.1 backbone to NaN.
"""),
    code("""
assert M.diagnose_forward(model, train_ds, cfg), (
    "Non-finite values in the forward pass — do not train. "
    "Check the first 'NO' row above."
)
"""),
    md("## 3 · Train"),
    code("""
model, train_info = M.train_model(
    model, train_ds, cfg,
    ckpt_path=CKPT,
    resume_from=RESUME,
    time_budget_h=TIME_BUDGET_H,
)
print("\\n", {k: v for k, v in train_info.items() if k != "history"})
if train_info["stopped_early"]:
    print("\\n!! Run incomplete. Re-run this notebook to resume from the checkpoint.")
if train_info.get("diverged") or train_info.get("nan_batches"):
    print("\\n!! TRAINING DIVERGED (non-finite loss). Results below are meaningless.")
    print("   Lower cfg.lr (try 5e-5) and re-run; delete the checkpoint first.")
"""),
    md("## 4 · Evaluate — accuracy on closed-end questions"),
    code("""
preds = M.generate_predictions(model, test_ds, cfg)
pred_answers = [M.parse_answer(p, cfg.strategy)[0] for p in preds]

gold    = [r["answer"] for r in test_recs]
options = [r["options"] for r in test_recs]
metrics = M.evaluate_closed_end(pred_answers, gold, options)

# A yes/no test set is rarely balanced. Accuracy alone cannot tell a model
# that learned the task from one that learned the class prior, so report
# the margin over always-guessing-the-majority explicitly.
metrics["MajorityBaseline"] = M.majority_baseline(test_recs)
metrics["LiftOverMajority"] = metrics["Accuracy"] - metrics["MajorityBaseline"]

# Did the visual gate actually open? It starts at ~0.047 by construction;
# if it has not moved, the images are being ignored and this is a text-only
# model in multimodal clothing.
metrics.update(M.measure_gate(model, test_ds, cfg))

# When a rationale is generated, score it against the gold MDMR too: a model
# can be right for the wrong reason, and the paper's whole claim is about
# interpretability, not just accuracy.
if cfg.strategy in ("Explanation", "Reasoning"):
    rats = [M.parse_answer(p, cfg.strategy)[1] for p in preds]
    for k, v in M.evaluate_open_end(rats, [r["solution"] for r in test_recs]).items():
        metrics[f"rationale/{k}"] = v

print(f"\\n=== {cfg.run_name()} ===")
for k, v in metrics.items():
    print(f"  {k:24s} {v:8.3f}")

lift = metrics["LiftOverMajority"]
print()
if lift <= 1.0:
    print(f"  !! Accuracy {metrics['Accuracy']:.1f}% vs majority-class "
          f"{metrics['MajorityBaseline']:.1f}% -- the model is predicting")
    print("     the class prior, not answering the question. Not usable.")
else:
    print(f"  +{lift:.1f} points over the majority-class baseline.")

print()
print(f"visual gate: lambda {metrics['gate/lambda_mean']:.4f} "
      f"(init 0.047, shift {metrics['gate/shift_from_init']:+.4f}, "
      f"max {metrics['gate/lambda_max']:.4f})")
if abs(metrics["gate/shift_from_init"]) < 0.01:
    print("  !! Visual gate has not moved from its 0.047 init -- the images")
    print("     are being ignored; a multimodal claim would be unsupported.")
print("\\npaper (English, closed-end Explanation): R-RAD 83.5 | R-SLAKE 86.3 | R-Path 87.0")
"""),
    md("## 5 · Inspect a few predictions"),
    code("""
for r, p in list(zip(test_recs, preds))[:5]:
    a, rat = M.parse_answer(p, cfg.strategy)
    snap = M.snap_to_choice(a, r["options"])
    print(f"Q    : {r['question']}")
    print(f"gold : {r['answer']}   pred: {a!r} -> snapped: {snap}  "
          f"{'OK' if snap == r['answer'] else 'WRONG'}")
    if rat:
        print(f"why  : {rat[:180]}...")
    print("-" * 90)
"""),
    md("## 6 · Save"),
    code("""
M.save_results(cfg, metrics, test_recs, preds, train_info)
print("\\nSave this notebook's output as a dataset (e.g. medthink-bn-runs) so")
print("notebook 05 can aggregate it with the other runs.")
"""),
])
write("02_train_closed_end.ipynb", nb02)


# --------------------------------------------------------------------------- #
# 03 - open-end
# --------------------------------------------------------------------------- #

nb03 = notebook([
    md("""
# 03 · Open-End Training — BLEU / ROUGE (paper Table 3)

Same architecture, open-ended questions. Scored with **Bengali-aware**
tokenisation, at two granularities:

| granularity | what it measures |
|---|---|
| `full/*`  | whole generation vs whole target (answer + rationale) — comparable to the paper's Table 3 |
| `ans/*`   | parsed answer vs gold answer — the clinically meaningful number |

Both are reported because the paper is ambiguous about which it used. Table 3
must be the *full* variant: R-SLAKE open-end answers average 1–2 tokens, which
admit no 4-grams at all, so its reported BLEU-4 of 10.3 cannot come from
answer-only scoring.

**Attach:** `medthink-bn-manifests`, `medthink-bn-features`.
"""),
    LIB_CELL,
    code(SETUP + """
import torch
from transformers import AutoTokenizer
print("cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""" + LOADER),
    md("## ⚙️ Config"),
    code("""
# ===========================================================================
# CHANGE THESE THREE LINES ONLY.
# ===========================================================================
PRESET   = "sanity"        # sanity | baseline | full
DATASET  = "rad"           # rad | slake | path
STRATEGY = "Explanation"   # without_R | Explanation | Reasoning  (used by "full")
# ===========================================================================
#   sanity   200 examples x 60 epochs, without_R   ~5 min, diagnostic only
#   baseline full data, without_R, paper epochs    the no-rationale reference
#   full     full data, STRATEGY above             the real experiment

_PRESETS = {
    "sanity":   dict(strategy="without_R", epochs=60,   quick_run=True),
    "baseline": dict(strategy="without_R", epochs=None, quick_run=False),
    "full":     dict(strategy=STRATEGY,    epochs=None, quick_run=False),
}
assert PRESET in _PRESETS, f"PRESET must be one of {list(_PRESETS)}"

cfg = M.Config(
    dataset    = DATASET,
    subset     = "open-end",
    **_PRESETS[PRESET],

    # Swap backbone here. Notebooks 00 and 01 do NOT need re-running.
    model_name = "csebuetnlp/banglat5",  # or "google/mt5-base" (then bs=4, ga=8)

    batch_size = 8,               # dynamic padding shrank sequences ~17x
    grad_accum = 4,               # 8 x 4 = effective batch 32, the paper's setting
    lr         = 1e-4,            # NOT the paper's 5e-4 -- see note below
    source_len = 512,
    target_len = 320,
    num_beams  = 1,
    seed       = 42,
)

TIME_BUDGET_H = 10.5
if cfg.quick_run and cfg.epochs is None:
    cfg.epochs = 2

assert cfg.subset == "open-end"
assert cfg.strategy in ["without_R", "Explanation", "Reasoning"]

n_train = 200 if cfg.quick_run else {"rad": 1241, "slake": 2976, "path": 9680}[cfg.dataset]
steps = n_train * cfg.resolved_epochs() // (cfg.batch_size * cfg.grad_accum)
print(f"PRESET={PRESET}  ->  {cfg.run_name()}")
print(f"  {n_train} examples x {cfg.resolved_epochs()} epochs = ~{steps} optimiser steps")
if PRESET == "sanity":
    print("  SANITY CHECK -- diagnostic only, do not report these numbers.")
print()
print(cfg.to_json())
CKPT = f"{WORK}/ckpt_{cfg.run_name()}.pt"

# /kaggle/working is wiped between sessions. If the previous session's
# output is attached, pull its checkpoint in so a long run resumes
# instead of silently restarting from epoch 0.
RESUME = M.find_checkpoint(CKPT, DATA_ROOT)
"""),
    md("""
## 1 · Data

Note the schema trap this handles: R-PathVQA open-end stores `answer` as a raw
string with a vestigial `["হ্যাঁ","না"]` in `choices`, while R-RAD and R-SLAKE
store an integer index into a one-element `choices`. Reading them the same way
silently yields "হ্যাঁ" as the gold answer for every PathVQA question.
"""),
    code(BUILD_DS),
    md("## 2 · Model"),
    code(MODEL_CELL),
    md("""
## 2b · Pre-flight check — is the forward pass finite?

A NaN loss produces a model that generates empty strings and scores 0.00 on
everything, which is easy to mistake for a bad result rather than a broken run.
This traces one forward pass and shows exactly where any non-finite value first
appears, before you spend hours training.

`mean gate lambda` should start near **0.05**. The visual pathway is
deliberately initialised almost closed so the randomly-initialised image
projection cannot overwrite the pretrained encoder at step 0 — that, combined
with the paper's lr of 5e-4, is what drives a T5 v1.1 backbone to NaN.
"""),
    code("""
assert M.diagnose_forward(model, train_ds, cfg), (
    "Non-finite values in the forward pass — do not train. "
    "Check the first 'NO' row above."
)
"""),
    md("## 3 · Train"),
    code("""
model, train_info = M.train_model(
    model, train_ds, cfg,
    ckpt_path=CKPT,
    resume_from=RESUME,
    time_budget_h=TIME_BUDGET_H,
)
print("\\n", {k: v for k, v in train_info.items() if k != "history"})
if train_info["stopped_early"]:
    print("\\n!! Run incomplete. Re-run to resume from the checkpoint.")
if train_info.get("diverged") or train_info.get("nan_batches"):
    print()
    print("!! TRAINING DIVERGED (non-finite loss). Results below are meaningless.")
    print("   Lower cfg.lr (try 5e-5) and re-run; delete the checkpoint first.")
"""),
    md("## 4 · Evaluate"),
    code("""
preds = M.generate_predictions(model, test_ds, cfg)

full_refs    = [M.build_target_text(r, cfg.strategy) for r in test_recs]
answer_preds = [M.parse_answer(p, cfg.strategy)[0] for p in preds]
answer_refs  = [r["answer"] for r in test_recs]

metrics = M.evaluate_open_end_dual(preds, full_refs, answer_preds, answer_refs)

# Open-end BLEU/ROUGE can look respectable purely from reproducing the answer
# style. Score always-emitting-the-most-common-training-answer as the floor
# a real result has to clear.
# A model that emits only the marker and stops still scores ~50 ROUGE-1 on a
# short without_R target, because the marker is half the reference. Measure
# that failure directly rather than letting it hide behind the score.
metrics.update(M.degenerate_rate(answer_preds))

const_ans, base = M.constant_answer_baseline(train_recs, test_recs)
metrics.update(base)

# Did the visual gate open? It starts at ~0.047 by construction; if it has
# not moved, the images are being ignored.
metrics.update(M.measure_gate(model, test_ds, cfg))

print(f"\\n=== {cfg.run_name()} ===")
print("\\nfull generation vs full target (comparable to paper Table 3)")
for k, v in metrics.items():
    if k.startswith("full/"):
        print(f"  {k[5:]:12s} {v:6.2f}")
print("\\nanswer span vs gold answer (clinical correctness)")
for k, v in metrics.items():
    if k.startswith("ans/"):
        print(f"  {k[4:]:12s} {v:6.2f}")


print(f"empty answers: {metrics['EmptyAnswerRate']:.1f}%  "
      f"(mean {metrics['MeanAnswerTokens']:.1f} tokens per answer)")
if metrics["EmptyAnswerRate"] > 20:
    print("  !! The model is emitting the marker and stopping. full/* scores")
    print("     are marker overlap, not answers -- read ans/* only.")
print()
print(f"constant-answer baseline (always {const_ans!r}):")
for m in ["ExactMatch", "TokenF1", "ROUGE-1"]:
    got, floor = metrics[f"ans/{m}"], metrics[f"baseline/{m}"]
    flag = "  <-- at or below the trivial floor" if got <= floor + 0.5 else ""
    print(f"  {m:12s} model {got:6.2f}   constant {floor:6.2f}{flag}")

print()
print(f"visual gate: lambda {metrics['gate/lambda_mean']:.4f} "
      f"(init 0.047, shift {metrics['gate/shift_from_init']:+.4f}, "
      f"max {metrics['gate/lambda_max']:.4f})")
if abs(metrics["gate/shift_from_init"]) < 0.01:
    print()
    print("  !! Visual gate has not moved from its 0.047 init -- the images")
    print("     are being ignored; a multimodal claim would be unsupported.")
print("\\npaper (English, Explanation): R-RAD R1 50.2 / B4 8.8 | "
      "R-SLAKE R1 53.1 / B4 9.9 | R-Path R1 41.9 / B4 5.8")
"""),
    md("## 5 · Inspect"),
    code("""
for r, p in list(zip(test_recs, preds))[:5]:
    a, rat = M.parse_answer(p, cfg.strategy)
    print(f"Q    : {r['question']}")
    print(f"gold : {r['answer']}")
    print(f"pred : {a}")
    if rat:
        print(f"why  : {rat[:180]}...")
    print("-" * 90)
"""),
    md("## 6 · Save"),
    code("M.save_results(cfg, metrics, test_recs, preds, train_info)"),
])
write("03_train_open_end.ipynb", nb03)


# --------------------------------------------------------------------------- #
# 04 - two-stage reasoning
# --------------------------------------------------------------------------- #

nb04 = notebook([
    md("""
# 04 · Two-Stage Reasoning

The paper's third strategy (Sec. 3.4): two independently trained models.

1. **Stage 1** — `(Q, I) → R`. Trained at the dataset's normal lr/epochs.
2. **Stage 2** — `(Q, I, R) → A`. Trained at **lr 5e-5 for 20 epochs**
   (paper Sec. 5.1).

Stage 2 consumes the rationale **generated by stage 1**, on both train and
test. Feeding gold rationales at training time and generated ones at test time
would create a train/test mismatch — and at test time gold rationales leak
the answer outright.

This notebook trains two models, so budget roughly double a single run.
"""),
    LIB_CELL,
    code(SETUP + """
import torch
from transformers import AutoTokenizer
print("cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""" + LOADER),
    md("## ⚙️ Config"),
    code("""
cfg1 = M.Config(
    dataset    = "rad",
    subset     = "closed-end",          # closed-end | open-end
    strategy   = "First-Stage_Reasoning",
    epochs     = None,                  # paper value for the dataset
    lr         = 1e-4,                  # not the paper's 5e-4: T5 v1.1 diverges
    batch_size = 8, grad_accum = 4,
    target_len = 320, seed = 42,
    quick_run  = True,                  # <<<< SET False FOR THE REAL RUN
)

TIME_BUDGET_H = 5.0                     # per stage; two stages in one session
if cfg1.quick_run:
    cfg1.epochs = 2

import dataclasses
cfg2 = dataclasses.replace(
    cfg1,
    strategy="Second-Stage_Reasoning",
    lr=5e-5,                            # paper Sec. 5.1
    epochs=2 if cfg1.quick_run else 20, # paper Sec. 5.1
    target_len=64,                      # stage 2 emits an answer only
)
print("stage 1:", cfg1.run_name(), "| stage 2:", cfg2.run_name())
"""),
    md("## 1 · Data"),
    code("""
tokenizer = AutoTokenizer.from_pretrained(cfg1.model_name, use_fast=False)
features, name_to_row = load_features()

train_recs = load_manifest(cfg1.dataset, cfg1.subset, "train")
test_recs  = load_manifest(cfg1.dataset, cfg1.subset, "test")
if cfg1.quick_run:
    train_recs = subsample(train_recs, 200, cfg1.seed)
    test_recs  = subsample(test_recs, 100, cfg1.seed)
print(f"train {len(train_recs)} | test {len(test_recs)}")
"""),
    md("## 2 · Stage 1 — (Q, I) → R"),
    code("""
M.set_seed(cfg1.seed)
s1_train = M.MedThinkDataset(train_recs, tokenizer, cfg1, features, name_to_row)
S1_CKPT = f"{WORK}/ckpt_{cfg1.run_name()}.pt"
S1_RESUME = M.find_checkpoint(S1_CKPT, DATA_ROOT)
model1 = M.load_model(cfg1, tokenizer)

assert M.diagnose_forward(model1, s1_train, cfg1), \
    "Non-finite forward pass — do not train. Check the first 'NO' row above."

model1, info1 = M.train_model(
    model1, s1_train, cfg1,
    ckpt_path=S1_CKPT,
    resume_from=S1_RESUME,
    time_budget_h=TIME_BUDGET_H,
)
print("\\nstage 1:", {k: v for k, v in info1.items() if k != "history"})
if info1.get("diverged") or info1.get("nan_batches"):
    print("!! STAGE 1 DIVERGED -- the rationales below are noise; stop here.")
if info1["stopped_early"]:
    raise RuntimeError(
        "Stage 1 hit TIME_BUDGET_H and is INCOMPLETE. Its checkpoint is saved -- "
        "re-run this notebook to resume stage 1. Continuing now would train "
        "stage 2 on rationales from a half-trained model."
    )
"""),
    md("### 2b · Generate rationales for train and test"),
    code("""
s1_test = M.MedThinkDataset(test_recs, tokenizer, cfg1, features, name_to_row)

# Generating rationales costs up to ~5h on R-PathVQA and is pure inference,
# so cache it: a session that dies in stage 2 must not have to redo it.
RAT_PATH = f"{WORK}/rationales_{cfg1.run_name()}.json"
_cached = ([RAT_PATH] if os.path.exists(RAT_PATH) else
           [str(x) for x in sorted(Path(DATA_ROOT).glob(f"**/rationales_{cfg1.run_name()}.json"))])

if _cached:
    rationales = json.load(open(_cached[0], encoding="utf-8"))
    print(f"reusing cached rationales from {_cached[0]} ({len(rationales)} entries)")
    rat_test = [rationales[M.record_uid(r)] for r in test_recs]
else:
    rat_train = M.generate_predictions(model1, s1_train, cfg1)
    rat_test  = M.generate_predictions(model1, s1_test,  cfg1)

    # Keyed by the globally unique id, NOT qid: R-PathVQA numbers train and test
    # from 1 independently, so 3257 qids collide and a qid-keyed dict would
    # overwrite ~35% of the training rationales with test-derived ones.
    rationales = {M.record_uid(r): t for r, t in zip(train_recs, rat_train)}
    rationales.update({M.record_uid(r): t for r, t in zip(test_recs, rat_test)})
    assert len(rationales) == len(train_recs) + len(test_recs), (
        "rationale key collision -- train rationales were overwritten by test ones"
    )
    with open(RAT_PATH, "w", encoding="utf-8") as fh:
        json.dump(rationales, fh, ensure_ascii=False)
    print(f"cached {len(rationales)} rationales -> {RAT_PATH}")

# Quality of the stage-1 rationales themselves.
r_metrics = M.evaluate_open_end(rat_test, [r["solution"] for r in test_recs])
print("stage-1 rationale quality vs gold MDMR:")
for k, v in r_metrics.items():
    print(f"  {k:12s} {v:6.2f}")

print("\\nexample:", rat_test[0][:250])
del model1
torch.cuda.empty_cache()
"""),
    md("## 3 · Stage 2 — (Q, I, R) → A"),
    code("""
M.set_seed(cfg2.seed)
s2_train = M.MedThinkDataset(train_recs, tokenizer, cfg2, features, name_to_row,
                             rationales=rationales)
s2_test  = M.MedThinkDataset(test_recs,  tokenizer, cfg2, features, name_to_row,
                             rationales=rationales)

print("--- stage-2 input ---\\n" +
      M.build_input_text(train_recs[0], cfg2.strategy, rationales[M.record_uid(train_recs[0])])[:400])

S2_CKPT = f"{WORK}/ckpt_{cfg2.run_name()}.pt"
S2_RESUME = M.find_checkpoint(S2_CKPT, DATA_ROOT)
model2 = M.load_model(cfg2, tokenizer)
model2, info2 = M.train_model(
    model2, s2_train, cfg2,
    lr=cfg2.lr, epochs=cfg2.epochs,
    ckpt_path=S2_CKPT,
    resume_from=S2_RESUME,
    time_budget_h=TIME_BUDGET_H,
)
print("\\nstage 2:", {k: v for k, v in info2.items() if k != "history"})
if info2.get("diverged") or info2.get("nan_batches"):
    print("!! STAGE 2 DIVERGED. Results below are meaningless.")
"""),
    md("## 4 · Evaluate"),
    code("""
preds = M.generate_predictions(model2, s2_test, cfg2)
pred_answers = [M.parse_answer(p, cfg2.strategy)[0] for p in preds]
gold = [r["answer"] for r in test_recs]

if cfg2.subset == "closed-end":
    metrics = M.evaluate_closed_end(pred_answers, gold, [r["options"] for r in test_recs])
    metrics["MajorityBaseline"] = M.majority_baseline(test_recs)
    metrics["LiftOverMajority"] = metrics["Accuracy"] - metrics["MajorityBaseline"]
    print("\\npaper (Two-Stage, closed-end): R-RAD 80.5 | R-SLAKE 79.1 | R-Path 87.2")
else:
    full_refs = [M.build_target_text(r, cfg2.strategy) for r in test_recs]
    metrics = M.evaluate_open_end_dual(preds, full_refs, pred_answers, gold)
    metrics.update(M.degenerate_rate(pred_answers))
    const_ans, base = M.constant_answer_baseline(train_recs, test_recs)
    metrics.update(base)

# Did the visual gate open? Reported for stage 2, the model that answers.
metrics.update(M.measure_gate(model2, s2_test, cfg2))

for k, v in metrics.items():
    print(f"  {k:24s} {v:6.2f}")

for k, v in r_metrics.items():
    metrics[f"stage1_rationale/{k}"] = v
"""),
    md("## 5 · Save"),
    code("""
info = {"stage1": info1, "stage2": info2}
M.save_results(cfg2, metrics, test_recs, preds, info)

with open(f"{WORK}/rationales_{cfg1.run_name()}.json", "w", encoding="utf-8") as fh:
    json.dump(rationales, fh, ensure_ascii=False)
print("saved generated rationales for reuse/inspection")
"""),
])
write("04_two_stage_reasoning.ipynb", nb04)


# --------------------------------------------------------------------------- #
# 05 - results
# --------------------------------------------------------------------------- #

nb05 = notebook([
    md("""
# 05 · Results — Tables & Figures

Aggregates every `metrics_*.json` produced by notebooks 02–04 into the tables
your paper needs, alongside the original English MedThink numbers.

**Attach:** the saved outputs of your training notebooks (e.g. `medthink-bn-runs`).
No GPU needed.
"""),
    LIB_CELL,
    code(SETUP + """
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
"""),
    md("## 1 · Collect runs"),
    code("""
paths = sorted(Path(DATA_ROOT).glob("**/metrics_*.json")) + \\
        sorted(Path(WORK).glob("metrics_*.json"))
print(f"found {len(paths)} run(s)")

runs = []
for p in paths:
    d = json.load(open(p, encoding="utf-8"))
    c, m, t = d["config"], d["metrics"], d.get("train", {})
    # Notebook 04 stores {'stage1': ..., 'stage2': ...}. Read flatly and a
    # truncated two-stage run reports stopped_early=False -- i.e. the
    # integrity gate would pass an incomplete run. Flatten it: the run is
    # incomplete if EITHER stage was cut short.
    if "stage1" in t:
        s1, s2 = t.get("stage1", {}), t.get("stage2", {})
        t = {
            "epochs_completed": s2.get("epochs_completed"),
            "epochs_target": s2.get("epochs_target"),
            "stopped_early": bool(s1.get("stopped_early") or s2.get("stopped_early")),
            "diverged": bool(s1.get("diverged") or s2.get("diverged")),
            "nan_batches": (s1.get("nan_batches", 0) or 0) + (s2.get("nan_batches", 0) or 0),
            "stage1_epochs": s1.get("epochs_completed"),
            "stage1_target": s1.get("epochs_target"),
        }
    row = {
        "model": c.get("model_name", "?").split("/")[-1],
        "dataset": c["dataset"], "subset": c["subset"], "strategy": c["strategy"],
        "epochs_done": t.get("epochs_completed"), "epochs_target": t.get("epochs_target"),
        "incomplete": t.get("stopped_early", False),
        "s1_epochs": t.get("stage1_epochs"),
        "diverged": t.get("diverged", False) or bool(t.get("nan_batches", 0)),
        "quick_run": c.get("quick_run", False),
    }
    row.update({k: round(v, 2) for k, v in m.items()})
    runs.append(row)

# Backbone is part of a run's identity, so mT5 and BanglaT5 results coexist
# instead of one silently replacing the other.
df = pd.DataFrame(runs).drop_duplicates(
    subset=["model", "dataset", "subset", "strategy"], keep="last")
df
"""),
    md("""
### Integrity check

A run that hit the time budget, or one left on `quick_run`, must never be
reported as a result. Flag them before they reach a table.
"""),
    code("""
bad = df[df["incomplete"] | df["quick_run"] | df.get("diverged", False)]
if len(bad):
    print("!! NOT publication-ready — these runs are truncated or quick_run:\\n")
    print(bad[["model", "dataset", "subset", "strategy", "epochs_done",
               "epochs_target", "incomplete", "quick_run", "diverged"]].to_string(index=False))
else:
    print("All runs completed their full epoch budget.")
"""),
    md("## 2 · Table 2 equivalent — closed-end accuracy"),
    code("""
PAPER_CLOSED = {   # English MedThink, paper Table 2
    ("rad", "without_R"): 79.0, ("rad", "Reasoning"): 73.9,
    ("rad", "Two-Stage"): 80.5, ("rad", "Explanation"): 83.5,
    ("slake", "without_R"): 82.5, ("slake", "Reasoning"): 80.8,
    ("slake", "Two-Stage"): 79.1, ("slake", "Explanation"): 86.3,
    ("path", "without_R"): 86.0, ("path", "Reasoning"): 83.1,
    ("path", "Two-Stage"): 87.2, ("path", "Explanation"): 87.0,
}

cl = df[(df["subset"] == "closed-end") & df["Accuracy"].notna()] if "Accuracy" in df else pd.DataFrame()
if len(cl):
    t = cl.pivot_table(index=["model", "dataset"], columns="strategy", values="Accuracy")
    print("Bengali (ours), closed-end accuracy %\\n")
    print(t.to_string())

    print("\\nvs English MedThink (paper Table 2)\\n")
    for mdl, ds in t.index:
        for st in t.columns:
            key = (ds, "Two-Stage" if "Two-Stage" in st or "Second" in st else st)
            if key in PAPER_CLOSED:
                ours, paper = t.loc[(mdl, ds), st], PAPER_CLOSED[key]
                if pd.notna(ours):
                    print(f"  {mdl:12s} {ds:6s} {st:24s} ours {ours:5.1f}  "
                          f"paper {paper:5.1f}  delta {ours - paper:+5.1f}")
else:
    print("No closed-end runs yet.")
"""),
    md("## 3 · Table 3 equivalent — open-end generation"),
    code("""
op = df[df["subset"] == "open-end"]
if len(op):
    full_cols = [c for c in op.columns if c.startswith("full/")]
    ans_cols  = [c for c in op.columns if c.startswith("ans/")]

    print("Full generation vs full target (comparable to paper Table 3)\\n")
    print(op[["model", "dataset", "strategy"] + full_cols].to_string(index=False))
    print("\\nAnswer only vs gold answer (clinical correctness)\\n")
    print(op[["model", "dataset", "strategy"] + ans_cols].to_string(index=False))
else:
    print("No open-end runs yet.")
"""),
    md("""
## 4 · Does the rationale help?

The paper's central claim is that adding MDMRs improves accuracy. This is that
delta, computed on your Bengali data — the headline number of your study.
"""),
    code("""
if len(cl):
    t = cl.pivot_table(index=["model", "dataset"], columns="strategy", values="Accuracy")
    if "without_R" in t.columns:
        print("accuracy delta vs the no-rationale baseline (percentage points)\\n")
        for st in [c for c in t.columns if c != "without_R"]:
            print(f"  {st}:")
            for idx in t.index:
                base, cur = t.loc[idx, "without_R"], t.loc[idx, st]
                if pd.notna(base) and pd.notna(cur):
                    print(f"      {idx[0]:12s} {idx[1]:6s} {cur - base:+5.1f}")
        print("\\npaper deltas (English): Explanation +4.5 / +3.8 / +1.0,")
        print("Two-Stage +1.5 / -3.4 / +1.2, Reasoning -5.1 / -1.7 / -2.9")
    else:
        print("Run the without_R baseline to compute this delta.")
"""),
    md("## 5 · Error analysis by anatomical region (paper Table 4)"),
    code("""
pred_files = sorted(Path(DATA_ROOT).glob("**/preds_*.jsonl")) + \\
             sorted(Path(WORK).glob("preds_*.jsonl"))
print(f"{len(pred_files)} prediction file(s)")

for pf in pred_files:
    if "closed-end" not in pf.name:
        continue
    recs = [json.loads(l) for l in open(pf, encoding="utf-8")]
    if not any(r.get("image_organ") for r in recs):
        continue
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in recs:
        organ = (r.get("image_organ") or "unknown").strip() or "unknown"
        snap = M.snap_to_choice(r["pred_answer"], r["options"])
        agg[organ][1] += 1
        agg[organ][0] += int(snap != r["gold_answer"])
    print(f"\\n{pf.name}")
    print(f"  {'region':22s}{'N':>6s}{'error %':>10s}")
    for organ, (err, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        if n >= 5:
            print(f"  {organ[:20]:22s}{n:>6d}{100*err/n:>10.2f}")
"""),
    md("## 6 · Export"),
    code("""
df.to_csv(f"{WORK}/all_results.csv", index=False)
with open(f"{WORK}/all_results.md", "w", encoding="utf-8") as fh:
    fh.write("# Bengali MedThink — results\\n\\n")
    fh.write(df.to_markdown(index=False))
print("wrote all_results.csv and all_results.md")
"""),
], accelerator="None")
write("05_results_report.ipynb", nb05)

print("\nAll notebooks generated in", OUT)
