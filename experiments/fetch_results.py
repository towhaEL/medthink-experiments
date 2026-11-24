"""
Download only the small result files (json/jsonl/log) of finished plan runs.

Each run also writes ~4 GB of checkpoints, so this never uses a plain
`kernels output`. Already-downloaded runs are skipped unless --force.

    python3 experiments/fetch_results.py --plan week1
    python3 experiments/fetch_results.py --plan week1 --only rad-closed-expl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from launch_runs import PLANS, run_state, slug_for    # noqa: E402  (also puts kaggle-fleet on sys.path)
from kaggle_fleet.kaggle_runner import KaggleRunner   # noqa: E402

RESULTS = ROOT / "experiments" / "results" / "v2"
PATTERN = r".*\.(json|jsonl|log|csv)$"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="One plan or several joined by commas")
    ap.add_argument("--only", help="Only runs whose slug contains this text")
    ap.add_argument("--force", action="store_true", help="Re-download existing runs")
    args = ap.parse_args()

    runner = KaggleRunner()
    am = runner.account_manager
    bad = 0
    for r in [r for p in args.plan.split(",") for r in PLANS[p]]:
        slug = slug_for(r)
        if args.only and args.only not in slug:
            continue
        out = RESULTS / slug
        if out.exists() and any(out.rglob("*.json")) and not args.force:
            print(f"[skip]  {slug} (already downloaded)")
            continue
        state = run_state(runner, slug, refresh=True)
        if state != "done":
            print(f"[{state:5s}] {slug}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        proc = am.run_kaggle(am.get_account(r["account"]),
                             ["kernels", "output", f"{r['account']}/{slug}", "-p", str(out),
                              "-o", "--file-pattern", PATTERN], timeout=600)
        n = sum(1 for _ in out.rglob("*") if _.is_file())
        ok = proc.returncode == 0 and n > 0
        bad += not ok
        print(f"[{'ok' if ok else 'FAIL':5s}] {slug}: {n} files")
        if not ok:
            print("        ", ((proc.stdout or "") + (proc.stderr or "")).strip()[-400:])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
