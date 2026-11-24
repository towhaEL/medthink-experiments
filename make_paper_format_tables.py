"""
make_paper_format_tables.py
===========================
Reproduce MedThink's Tables 2, 3 and 4 (Findings of NAACL 2025) using the
Bengali results, in the paper's own layout so they drop straight into a
write-up next to the originals.

  Table 2  Accuracy (%) on closed-end questions, methods as rows, datasets as
           columns, each rationale row annotated with its delta against the
           method's own no-rationale row -- exactly the paper's convention.
  Table 3  Rouge-1/2/L and BLEU-1..4 on open-end questions, grouped by dataset.
           The paper scores the FULL generation (see note in the output), so
           full/* is used for comparability; an answer-only companion follows.
  Table 4  Per-region error rate on R-SLAKE closed-end. Needs the per-example
           preds_*.jsonl files, not just the metrics summaries.

Usage:  python make_paper_format_tables.py
"""

import glob
import io
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "metrics_json")
PRED_DIRS = [os.path.join(HERE, d) for d in ("pred", "preds", "metrics_json", "outputs")]
PRED_EXT = (".jsonl", ".json", ".txt")   # the export has landed under all three
OUT = os.path.join(HERE, "paper_tables")
os.makedirs(OUT, exist_ok=True)

DS = ["rad", "slake", "path"]
DSN = {"rad": "R-RAD", "slake": "R-SLAKE", "path": "R-Path"}
ROWS = [("without_R", "w/o R"), ("Reasoning", "w/ Reasoning"),
        ("Second-Stage_Reasoning", "w/ Two-Stage Reasoning"),
        ("Explanation", "w/ Explanation")]

# MedThink's own English numbers, verbatim from the paper.
EN_T2 = {"without_R": {"rad": 79.0, "slake": 82.5, "path": 86.0},
         "Reasoning": {"rad": 73.9, "slake": 80.8, "path": 83.1},
         "Second-Stage_Reasoning": {"rad": 80.5, "slake": 79.1, "path": 87.2},
         "Explanation": {"rad": 83.5, "slake": 86.3, "path": 87.0}}
EN_T3 = {  # Rouge-1/2/L, BLEU-1..4
    ("rad", "Reasoning"): [49.8, 20.3, 29.3, 37.8, 22.7, 14.0, 8.9],
    ("rad", "Second-Stage_Reasoning"): [49.1, 19.9, 28.7, 37.7, 22.5, 13.9, 8.8],
    ("rad", "Explanation"): [50.2, 20.2, 29.5, 38.3, 22.9, 14.0, 8.8],
    ("slake", "Reasoning"): [53.5, 22.8, 32.1, 39.5, 24.3, 15.5, 10.0],
    ("slake", "Second-Stage_Reasoning"): [53.2, 23.1, 32.0, 39.5, 24.5, 15.8, 10.3],
    ("slake", "Explanation"): [53.1, 22.7, 31.7, 39.2, 24.1, 15.4, 9.9],
    ("path", "Reasoning"): [41.5, 13.0, 24.8, 31.8, 17.0, 9.6, 5.7],
    ("path", "Second-Stage_Reasoning"): [41.7, 13.2, 24.9, 32.1, 17.1, 9.7, 5.8],
    ("path", "Explanation"): [41.9, 13.2, 25.0, 32.1, 17.1, 9.7, 5.8],
}


# R-SLAKE tags the same anatomy under several spellings, Bengali and English.
# Merging them reproduces the paper's exact region counts (141/141/91/16/5/22).
REGION = {
    "ফুসফুস": "Lung",
    "পেট": "Abdomen", "উদর": "Abdomen",
    "পেট (অ্যাবডোমেন)": "Abdomen", "অ্যাবডোমেন": "Abdomen",
    "ব্রেইন_টিস্যু": "Head", "মস্তিষ্ক_কলা": "Head", "মস্তিষ্ক_মুখমণ্ডল": "Head",
    "Brain_Tissue": "Head", "মস্তিষ্কের টিস্যু": "Head", "মস্তিষ্ক": "Head",
    "ঘাড়": "Neck", "Neck": "Neck",
    "বক্ষ_মিডিয়াস্টিনাল": "Chest", "Chest_mediastinal": "Chest",
    "Pelvic Cavity": "Pelvic Cavity", "পেলভিক ক্যাভিটি": "Pelvic Cavity",
}
PAPER_T4 = {"Lung": (12.06, 9.93), "Abdomen": (24.11, 19.15), "Head": (18.68, 13.18),
            "Neck": (18.75, 6.25), "Chest": (20.00, 0.00), "Pelvic Cavity": (4.55, 13.64)}
ORDER_T4 = ["Lung", "Abdomen", "Head", "Neck", "Chest", "Pelvic Cavity"]


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        d = json.load(io.open(f, encoding="utf-8"))
        c, t = d["config"], d.get("train", {})
        if "stage1" in t:
            s1, s2 = t.get("stage1", {}), t.get("stage2", {})
            inc = bool(s1.get("stopped_early") or s2.get("stopped_early"))
        else:
            inc = bool(t.get("stopped_early"))
        out[(c["dataset"], c["subset"], c["strategy"])] = (d["metrics"], inc)
    return out


R = load()
lines = []
w = lines.append


def num(v, nd=1):
    return "--" if v is None or v != v else f"{v:.{nd}f}"


# ------------------------------------------------------------------ Table 2 --
w("## Table 2 — Accuracy (%) on Closed-End Questions\n")
w("Deltas are against each method's own w/o-R row, following the paper.\n")
w("| Methods | R-RAD | R-SLAKE | R-Path | Parameters |")
w("|:---|---:|---:|---:|:---|")
for strat, label in ROWS:
    cells = []
    for d in DS:
        v = EN_T2[strat][d]
        cells.append(f"{v:.1f}" if strat == "without_R"
                     else f"{v:.1f} ({v - EN_T2['without_R'][d]:+.1f})")
    w(f"| MedThink ({label}) *(English, paper)* | " + " | ".join(cells) + " | 0.2B |")
w("| | | | | |")
base = {d: R[(d, "closed-end", "without_R")][0]["Accuracy"] for d in DS}
for strat, label in ROWS:
    cells = []
    for d in DS:
        m, inc = R[(d, "closed-end", strat)]
        a = m["Accuracy"]
        s = f"{a:.1f}" if strat == "without_R" else f"{a:.1f} ({a - base[d]:+.1f})"
        cells.append(s + ("†" if inc else ""))
    w(f"| **Ours-Bengali ({label})** | " + " | ".join(cells) + " | 0.25B |")
w("| Majority-class baseline | " +
  " | ".join(f"{R[(d,'closed-end','without_R')][0]['MajorityBaseline']:.1f}" for d in DS) +
  " | -- |")
w("\n† run did not reach its full epoch budget.\n")

# ------------------------------------------------------------------ Table 3 --
COLS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
for tag, pre, note in [
    ("3", "full/", "Whole generation vs whole target -- the variant the paper's "
                   "Table 3 reports (its BLEU-4 of ~10 is unreachable from 1-2 token "
                   "answers alone)."),
    ("3b", "ans/", "Parsed answer vs gold answer -- the clinically meaningful "
                   "variant. Not directly comparable to the paper."),
]:
    w(f"\n## Table {tag} — Score (%) on Open-End Questions ({pre}*)\n")
    w(note + "\n")
    w("| Dataset | Strategy | Rouge-1 | Rouge-2 | Rouge-L | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |")
    w("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
    for d in DS:
        w(f"| **{DSN[d]}** | | | | | | | | |")
        for strat, label in ROWS:
            key = (d, "open-end", strat)
            if key not in R:
                continue
            m, inc = R[key]
            vals = " | ".join(num(m.get(pre + c)) for c in COLS)
            nm = label.replace("w/ ", "").replace("w/o R", "w/o R")
            w(f"| | {nm}{'†' if inc else ''} | {vals} |")
            if tag == "3" and (d, strat) in EN_T3:
                en = " | ".join(f"*{v:.1f}*" for v in EN_T3[(d, strat)])
                w(f"| | *{nm} (English, paper)* | {en} |")
    w("\n† run did not reach its full epoch budget.\n")

# ------------------------------------------------------------------ Table 4 --
w("\n## Table 4 — Error rate per anatomical region, R-SLAKE closed-end (lower is better)\n")
pred = {}
for strat, key in [("without_R", "without_R"), ("Explanation", "Explanation")]:
    for dd in PRED_DIRS:
        for ext in PRED_EXT:
            f = os.path.join(dd, f"preds_slake_closed-end_{key}_banglat5{ext}")
            if os.path.exists(f):
                pred[strat] = f
                break
        if strat in pred:
            break

if len(pred) < 2:
    w("**Not generated — requires the per-example prediction files.**\n")
    w("Expected `preds_slake_closed-end_{without_R,Explanation}_banglat5.jsonl` "
      "in `pred/`.\n")
else:
    import sys
    sys.path.insert(0, HERE)
    import medthink_bn as M

    agg = defaultdict(lambda: {"n": 0, "without_R": 0, "Explanation": 0})
    unmapped = set()
    for strat, f in pred.items():
        for line in io.open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            raw = (r.get("image_organ") or "").strip()
            organ = REGION.get(raw)
            if organ is None:
                unmapped.add(raw)
                continue
            # Free-form generation is snapped to the nearest option before
            # scoring, exactly as the accuracy metric does.
            snap = M.snap_to_choice(r["pred_answer"], r["options"])
            if strat == "without_R":
                agg[organ]["n"] += 1
            agg[organ][strat] += int(snap != r["gold_answer"])

    w("| Anatomical Regions (Number) | w/o Rationale ↓ | Explanation ↓ | "
      "*paper w/o R* | *paper Expl.* |")
    w("|:---|---:|---:|---:|---:|")
    for organ in ORDER_T4:
        v = agg.get(organ)
        if not v or not v["n"]:
            continue
        pw, pe = PAPER_T4[organ]
        w(f"| {organ} (N={v['n']}) | {100*v['without_R']/v['n']:.2f}% "
          f"| {100*v['Explanation']/v['n']:.2f}% | *{pw:.2f}%* | *{pe:.2f}%* |")
    tot = sum(v["n"] for v in agg.values())
    ew = sum(v["without_R"] for v in agg.values())
    ee = sum(v["Explanation"] for v in agg.values())
    w(f"| **All (N={tot})** | **{100*ew/tot:.2f}%** | **{100*ee/tot:.2f}%** | -- | -- |")
    if unmapped:
        w(f"\nUnmapped region labels (excluded): {sorted(unmapped)}")
    w("\nRegion labels appear in both Bengali and English spellings in R-SLAKE; "
      "they are merged to the paper's six regions, which reproduces its exact "
      "counts (141/141/91/16/5/22).\n")

text = ("# Bengali MedThink — results in the paper's table format\n\n"
        "Backbone `csebuetnlp/banglat5` (247M) replacing UnifiedQA/T5-base; "
        "DETR-101 visual encoder (100x256); lr 1e-4 (not the paper's 5e-4, which "
        "diverges on a T5 v1.1 backbone); effective batch 32; greedy decoding.\n\n"
        + "\n".join(lines) + "\n")
p = os.path.join(OUT, "paper_format_tables.md")
io.open(p, "w", encoding="utf-8").write(text)
print(text)
print(f"\nwritten -> {p}")
