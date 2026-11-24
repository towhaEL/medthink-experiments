"""
analyze_conditions.py - every v2b condition (all plans), grouped over seeds, + key McNemar tests.

    python3 experiments/analyze_conditions.py      # prints Markdown

A "condition" is the slug with its seed removed, e.g. mtbn-v2b-slake-closed-expl-s43-aw50-bmc
-> slake | expl-aw50-bmc. Covers E1/E2/E3b/E3c/E5/Path. Pure stdlib; reuses analyze_v2b helpers.
"""
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_v2b import RES, load, mcnemar  # noqa: E402

PAT = re.compile(r"^mtbn-v2b-(rad|slake|path)-closed-(\w+?)-s(\d+)(.*)$")


def collect():
    runs = defaultdict(dict)  # (ds, cond) -> {seed: (metrics, correct)}
    for d in sorted(RES.glob("mtbn-v2b-*")):
        m = PAT.match(d.name)
        if not m or m.group(2) == "sanity":
            continue
        r = load(d.name)
        if r:
            ds, tag, seed, rest = m.groups()
            runs[(ds, tag + rest)][int(seed)] = r
    return runs


def pooled(runs, ds, a, b):
    A, B = runs.get((ds, a), {}), runs.get((ds, b), {})
    t01 = t10 = 0
    seeds = sorted(A.keys() & B.keys())
    for s in seeds:
        n01, n10, _ = mcnemar(A[s][1], B[s][1])
        t01 += n01
        t10 += n10
    n = t01 + t10
    if not n:
        return None
    k = min(t01, t10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return seeds, t01, t10, p


def main():
    runs = collect()
    out = ["## All conditions (test accuracy; ablation and gate are means over seeds)\n",
           "| Dataset | Condition | n | Acc mean ± std | per seed | zero img | shuffled | real − shuf | λ |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (ds, cond) in sorted(runs):
        rs = runs[(ds, cond)]
        accs = [rs[s][0]["metrics"]["Accuracy"] for s in sorted(rs)]
        g = lambda k: st.mean(rs[s][0]["metrics"].get(k, float("nan")) for s in rs)
        sd = f" ± {st.stdev(accs):.1f}" if len(accs) > 1 else ""
        real, zero, shuf = g("ablation/real/ChoiceAccuracy"), g("ablation/zero/ChoiceAccuracy"), g("ablation/shuffle/ChoiceAccuracy")
        out.append(f"| {ds.upper()} | {cond} | {len(accs)} | {st.mean(accs):.1f}{sd} | "
                   + " / ".join(f"{a:.1f}" for a in accs)
                   + f" | {zero:.1f} | {shuf:.1f} | {real - shuf:+.1f} | {g('gate/lambda_mean'):.3f} |")

    out += ["\n## Pooled paired McNemar (A vs B; seeds present in both)\n",
            "| Dataset | A | B | seeds | B right & A wrong | A right & B wrong | p |", "|---|---|---|---|---|---|---|"]
    pairs = [("wor", "expl-aw50-bmc"), ("wor-bmc", "expl-aw50-bmc"), ("expl-aw50", "expl-aw50-bmc"),
             ("wor", "wor-bmc"), ("expl-bmc", "expl-aw50-bmc"), ("wor", "expl"), ("wor", "expl-aw50"),
             ("wor-en", "expl-en"), ("wor-en", "expl-aw50-en")]
    for ds in ("rad", "slake", "path"):
        for a, b in pairs:
            r = pooled(runs, ds, a, b)
            if r:
                seeds, t01, t10, p = r
                out.append(f"| {ds.upper()} | {a} | {b} | {','.join(map(str, seeds))} | {t01} | {t10} | {p:.3g} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
