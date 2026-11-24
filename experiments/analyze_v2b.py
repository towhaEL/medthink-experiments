"""
analyze_v2b.py - 3-seed closed-end table + paired McNemar tests for protocol v2b.

    python3 experiments/analyze_v2b.py            # prints Markdown to stdout

Reads experiments/results/v2/mtbn-v2b-*/{metrics,preds}_*. Pure stdlib.
"""
import glob
import json
import math
import statistics as st
from pathlib import Path

RES = Path(__file__).parent / "results" / "v2"
SEEDS = (42, 43, 44)


def load(slug):
    d = RES / slug
    m = glob.glob(str(d / "metrics_*.json"))
    p = glob.glob(str(d / "preds_*.jsonl"))
    if not (m and p):
        return None
    metrics = json.load(open(m[0], encoding="utf-8"))
    preds = [json.loads(l) for l in open(p[0], encoding="utf-8")]
    correct = {(x["qid"], x.get("image_key")): x["pred_answer"].strip() == x["gold_answer"].strip()
               for x in preds}
    return metrics, correct


def mcnemar(a, b):
    """Exact two-sided McNemar on paired correctness dicts. Returns (b_only, a_only, p)."""
    keys = a.keys() & b.keys()
    n01 = sum(1 for k in keys if not a[k] and b[k])     # b right, a wrong
    n10 = sum(1 for k in keys if a[k] and not b[k])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def slug(ds, tag, seed, aw=None):
    s = f"mtbn-v2b-{ds}-closed-{tag}-s{seed}"
    return s + (f"-aw{round(aw * 100):02d}" if aw is not None else "")


def ms(xs):
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.1f} (n=1)"
    return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f} (n={len(xs)})"


def main():
    out = []
    conds = [("w/o-R", "wor", None), ("Explanation", "expl", None), ("Expl + α=0.5", "expl", 0.5)]
    out.append("## Main table (test accuracy, mean ± std over seeds)\n")
    out.append("| Dataset | " + " | ".join(c[0] for c in conds) + " | α=0.5 − w/o-R |")
    out.append("|---|" + "---|" * (len(conds) + 1))
    runs = {}
    for ds in ("rad", "slake"):
        row = []
        for name, tag, aw in conds:
            accs = []
            for s in SEEDS:
                r = load(slug(ds, tag, s, aw))
                if r:
                    runs[(ds, name, s)] = r
                    accs.append(r[0]["metrics"]["Accuracy"])
            row.append(accs)
        gap = (st.mean(row[2]) - st.mean(row[0])) if row[0] and row[2] else float("nan")
        out.append(f"| {ds.upper()} | " + " | ".join(ms(a) for a in row) + f" | {gap:+.1f} |")

    out.append("\n## Paired McNemar (exact, same seed, same test questions)\n")
    out.append("| Dataset | Comparison | Seed | B right / A wrong | A right / B wrong | p |")
    out.append("|---|---|---|---|---|---|")
    pairs = [("w/o-R", "Explanation"), ("w/o-R", "Expl + α=0.5"), ("Explanation", "Expl + α=0.5")]
    for ds in ("rad", "slake"):
        for a, b in pairs:
            tot01 = tot10 = 0
            for s in SEEDS:
                ra, rb = runs.get((ds, a, s)), runs.get((ds, b, s))
                if not (ra and rb):
                    continue
                n01, n10, p = mcnemar(ra[1], rb[1])
                tot01 += n01
                tot10 += n10
                out.append(f"| {ds.upper()} | A={a} vs B={b} | {s} | {n01} | {n10} | {p:.3g} |")
            n = tot01 + tot10
            if n:
                k = min(tot01, tot10)
                p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
                out.append(f"| {ds.upper()} | A={a} vs B={b} | pooled | {tot01} | {tot10} | {p:.3g} |")

    out.append("\n## α sweep, seed 42 (test accuracy)\n")
    out.append("| Dataset | Expl (no α) | α=0.3 | α=0.5 | α=0.8 | α=0.95 | w/o-R |")
    out.append("|---|---|---|---|---|---|---|")
    for ds in ("rad", "slake"):
        cells = []
        for tag, aw in [("expl", None), ("expl", 0.3), ("expl", 0.5), ("expl", 0.8), ("expl", 0.95), ("wor", None)]:
            r = load(slug(ds, tag, 42, aw))
            cells.append(f"{r[0]['metrics']['Accuracy']:.1f}" if r else "⏳")
        out.append(f"| {ds.upper()} | " + " | ".join(cells) + " |")

    out.append("\n## Image reliance and gate (mean over seeds)\n")
    out.append("| Dataset | Condition | real | zero image | shuffled image | real − shuffled | gate λ mean | best / run epochs |")
    out.append("|---|---|---|---|---|---|---|---|")
    for ds in ("rad", "slake"):
        for name, _, _ in conds:
            rs = [runs[(ds, name, s)][0] for s in SEEDS if (ds, name, s) in runs]
            if not rs:
                continue
            g = lambda k: st.mean(r["metrics"].get(k, float("nan")) for r in rs)
            real, zero, shuf = g("ablation/real/ChoiceAccuracy"), g("ablation/zero/ChoiceAccuracy"), g("ablation/shuffle/ChoiceAccuracy")
            ep = ", ".join(f"{r['train']['best_epoch']}/{r['train']['epochs_completed']}" for r in rs)
            out.append(f"| {ds.upper()} | {name} | {real:.1f} | {zero:.1f} | {shuf:.1f} | {real - shuf:+.1f} | {g('gate/lambda_mean'):.3f} | {ep} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
