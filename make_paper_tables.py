"""
make_paper_tables.py
====================
Turn the 60-column dump from notebook 05 into the three tables a paper needs.

Dropped on purpose:
  * model / quick_run / diverged  -- constant across runs; state in the caption
  * Unresolved                    -- 0.00 everywhere except one 0.37
  * gate/lambda_max               -- a per-token maximum; the mean is the claim
  * baseline/BLEU-*               -- UNRELIABLE. A one-token constant answer has
                                     no 2/3/4-grams, and the smoothing in the
                                     run that produced these scored the missing
                                     orders as perfect, so BLEU rose with n
                                     (path: 0.65 -> 3.85 -> 6.98 -> 9.40).
                                     Fixed in the library since; use the
                                     ExactMatch / TokenF1 / ROUGE-1 floors.
  * ans/BLEU-3, ans/BLEU-4        -- near-degenerate on 1-2 token answers
  * rationale/* duplicated as stage1_rationale/* -- merged into one column

Usage:  python make_paper_tables.py
"""

import glob
import io
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "metrics_json")
OUT = os.path.join(HERE, "paper_tables")
os.makedirs(OUT, exist_ok=True)

# MedThink's English results, paper Tables 2 and 3.
PAPER_CLOSED = {
    ("rad", "without_R"): 79.0, ("rad", "Reasoning"): 73.9,
    ("rad", "Second-Stage_Reasoning"): 80.5, ("rad", "Explanation"): 83.5,
    ("slake", "without_R"): 82.5, ("slake", "Reasoning"): 80.8,
    ("slake", "Second-Stage_Reasoning"): 79.1, ("slake", "Explanation"): 86.3,
    ("path", "without_R"): 86.0, ("path", "Reasoning"): 83.1,
    ("path", "Second-Stage_Reasoning"): 87.2, ("path", "Explanation"): 87.0,
}
PAPER_OPEN_R1 = {   # full-generation ROUGE-1, paper Table 3
    ("rad", "Explanation"): 50.2, ("rad", "Reasoning"): 49.8,
    ("rad", "Second-Stage_Reasoning"): 49.1,
    ("slake", "Explanation"): 53.1, ("slake", "Reasoning"): 53.5,
    ("slake", "Second-Stage_Reasoning"): 53.2,
    ("path", "Explanation"): 41.9, ("path", "Reasoning"): 41.5,
    ("path", "Second-Stage_Reasoning"): 41.7,
}
DS = {"rad": "R-RAD", "slake": "R-SLAKE", "path": "R-Path"}
ST = {"without_R": "w/o rationale", "Explanation": "Explanation",
      "Reasoning": "Reasoning", "Second-Stage_Reasoning": "Two-Stage"}
ORDER = ["without_R", "Explanation", "Reasoning", "Second-Stage_Reasoning"]


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        d = json.load(io.open(f, encoding="utf-8"))
        c, m, t = d["config"], d["metrics"], d.get("train", {})
        if "stage1" in t:                       # notebook 04 nests its two stages
            s1, s2 = t.get("stage1", {}), t.get("stage2", {})
            done, target = s1.get("epochs_completed"), s1.get("epochs_target")
            early = bool(s1.get("stopped_early") or s2.get("stopped_early"))
        else:
            done, target = t.get("epochs_completed"), t.get("epochs_target")
            early = bool(t.get("stopped_early"))
        r = {"dataset": c["dataset"], "subset": c["subset"], "strategy": c["strategy"],
             "epochs": done, "epochs_target": target, "incomplete": early}
        # the rationale-quality columns are the same measurement under two names
        for k in ("ROUGE-1", "ROUGE-2", "BLEU-4"):
            r[f"rat_{k}"] = m.get(f"rationale/{k}", m.get(f"stage1_rationale/{k}"))
        r.update(m)
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(v, nd=2):
    return "" if v is None or v != v else round(float(v), nd)


df = load()

# --------------------------------------------------------------------- A ---
# Closed-end accuracy: the paper's Table 2.
cl = df[df.subset == "closed-end"].copy()
base = {d: cl[(cl.dataset == d) & (cl.strategy == "without_R")].Accuracy.iloc[0]
        for d in cl.dataset.unique()}
A = []
for d in ["rad", "slake", "path"]:
    for s in ORDER:
        row = cl[(cl.dataset == d) & (cl.strategy == s)]
        if row.empty:
            continue
        r = row.iloc[0]
        A.append({
            "Dataset": DS[d],
            "Strategy": ST[s],
            "Accuracy": fmt(r.Accuracy),
            "Delta_vs_noRationale": "" if s == "without_R" else fmt(r.Accuracy - base[d]),
            "MajorityBaseline": fmt(r.MajorityBaseline),
            "LiftOverMajority": fmt(r.LiftOverMajority),
            "MedThink_English": PAPER_CLOSED.get((d, s), ""),
            "Delta_vs_English": fmt(r.Accuracy - PAPER_CLOSED[(d, s)]) if (d, s) in PAPER_CLOSED else "",
            "RationaleROUGE1": fmt(r.get("rat_ROUGE-1")),
            "GateLambda": fmt(r["gate/lambda_mean"], 3),
            "Epochs": f"{int(r.epochs)}/{int(r.epochs_target)}" if pd.notna(r.epochs) else "",
        })
tA = pd.DataFrame(A)

# --------------------------------------------------------------------- B ---
# Open-end: answer-level correctness is the honest number; full/* is what the
# paper's Table 3 actually measures, so both are reported.
op = df[df.subset == "open-end"].copy()
baseo = {d: op[(op.dataset == d) & (op.strategy == "without_R")]["ans/ExactMatch"].iloc[0]
         for d in op.dataset.unique()}
# Constant-answer floors, RECOMPUTED with the corrected BLEU (see module docstring).
FLOOR_EM = {"rad": 4.47, "slake": 6.05, "path": 2.58}
B = []
for d in ["rad", "slake", "path"]:
    for s in ORDER:
        row = op[(op.dataset == d) & (op.strategy == s)]
        if row.empty:
            continue
        r = row.iloc[0]
        B.append({
            "Dataset": DS[d],
            "Strategy": ST[s],
            "Ans_ExactMatch": fmt(r["ans/ExactMatch"]),
            "Delta_vs_noRationale": "" if s == "without_R" else fmt(r["ans/ExactMatch"] - baseo[d]),
            "ConstantFloor_EM": FLOOR_EM[d],
            "Ans_TokenF1": fmt(r["ans/TokenF1"]),
            "Ans_ROUGE1": fmt(r["ans/ROUGE-1"]),
            "Ans_ROUGEL": fmt(r["ans/ROUGE-L"]),
            "Ans_BLEU1": fmt(r["ans/BLEU-1"]),
            "Ans_BLEU2": fmt(r["ans/BLEU-2"]),
            "Full_ROUGE1": fmt(r["full/ROUGE-1"]),
            "Full_BLEU4": fmt(r["full/BLEU-4"]),
            "MedThink_English_FullROUGE1": PAPER_OPEN_R1.get((d, s), ""),
            "EmptyAnswerRate": fmt(r.get("EmptyAnswerRate")),
            "GateLambda": fmt(r["gate/lambda_mean"], 3),
            "Epochs": f"{int(r.epochs)}/{int(r.epochs_target)}" if pd.notna(r.epochs) else "",
        })
tB = pd.DataFrame(B)

# --------------------------------------------------------------------- C ---
# The headline: does adding a rationale help, in Bengali vs English?
PAPER_DELTA = {   # paper Table 2, relative to its own w/o-rationale row
    ("rad", "Explanation"): +4.5, ("rad", "Reasoning"): -5.1,
    ("rad", "Second-Stage_Reasoning"): +1.5,
    ("slake", "Explanation"): +3.8, ("slake", "Reasoning"): -1.7,
    ("slake", "Second-Stage_Reasoning"): -3.4,
    ("path", "Explanation"): +1.0, ("path", "Reasoning"): -2.9,
    ("path", "Second-Stage_Reasoning"): +1.2,
}
C = []
for s in ["Explanation", "Reasoning", "Second-Stage_Reasoning"]:
    row = {"Strategy": ST[s]}
    for d in ["rad", "slake", "path"]:
        got = cl[(cl.dataset == d) & (cl.strategy == s)]
        row[f"{DS[d]}_Bengali"] = fmt(got.Accuracy.iloc[0] - base[d]) if not got.empty else ""
        row[f"{DS[d]}_English"] = PAPER_DELTA.get((d, s), "")
    C.append(row)
tC = pd.DataFrame(C)

for name, t in [("table2_closed_end", tA), ("table3_open_end", tB),
                ("table4_rationale_effect", tC)]:
    t.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    print(f"  {name}.csv  ({len(t)} rows, {len(t.columns)} cols)")

with io.open(os.path.join(OUT, "paper_tables.md"), "w", encoding="utf-8") as fh:
    fh.write("# Bengali MedThink — paper tables\n\n")
    fh.write("Backbone: `csebuetnlp/banglat5` (247M). Visual encoder: DETR-101 "
             "(100x256). lr 1e-4, effective batch 32, greedy decoding.\n\n")
    fh.write("## Table 2 — Closed-end accuracy (%)\n\n")
    fh.write(tA.to_markdown(index=False) + "\n\n")
    fh.write("## Table 3 — Open-end generation\n\n")
    fh.write(tB.to_markdown(index=False) + "\n\n")
    fh.write("## Table 4 — Effect of rationales, Bengali vs English (accuracy points "
             "vs the no-rationale baseline)\n\n")
    fh.write(tC.to_markdown(index=False) + "\n")
print("  paper_tables.md")
