"""
Generate per-run copies of notebook 02 and push them to Kaggle via KaggleFleet
(../kaggle-fleet, state in ~/.kaggle_fleet).

Each run is notebook 02 with its config cell filled in, pushed as its own
private kernel so runs never overwrite each other's versions or outputs.

    python3 experiments/launch_runs.py --plan sanity --dry-run
    python3 experiments/launch_runs.py --plan sanity
    python3 experiments/launch_runs.py --plan week1
    python3 experiments/launch_runs.py --plan week1 --schedule   # keep <=2 per account

Kaggle allows 2 concurrent GPU sessions per account and rejects further
pushes, so --schedule keeps each account at that limit and pushes the next
run when a slot frees. It never re-pushes a run that is running, finished,
or genuinely failed on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# KaggleFleet (the packaged successor of kaggle_hub) owns accounts and job history.
# Use an installed copy if present, else the sibling checkout.
_FLEET_SRC = ROOT.parent / "kaggle-fleet" / "src"
if _FLEET_SRC.is_dir():
    sys.path.insert(0, str(_FLEET_SRC))

from kaggle_fleet.kaggle_runner import KaggleRunner        # noqa: E402
from kaggle_fleet.models import JobStatus, RunOptions      # noqa: E402

TEMPLATE = ROOT / "notebooks" / "02_train_closed_end.ipynb"
OUT_DIR = ROOT / "experiments" / "generated"
DATASETS = ["towhaelahi/medthink-bn-features", "towhaelahi/medthink-bn-manifests"]
# Extra Kaggle dataset per non-default visual feature set (nb 01b output).
FEATURE_DATASETS = {"biomedclip": "towhaelahi/medthink-bn-biomedclip"}
FEATURE_TAG = {"biomedclip": "bmc"}
# E5 English control: backbone + extra dataset holding the English JSONs.
LANG_MODEL = {"en": "allenai/unifiedqa-t5-base"}
LANG_DATASET = {"en": "towhaelahi/medthink-dataset"}
STRAT_TAG = {"without_R": "wor", "Explanation": "expl", "Reasoning": "reas"}


def run(account, dataset, strategy, seed, preset="full", prefix="mtbn-v2b", **extra):
    """One run. `extra` may set ans_weight / gate_bias / vision_warmup / features (AARL switches),
    lang ("en" = E5 English control with UnifiedQA), model, lr."""
    preset = "baseline" if strategy == "without_R" and preset == "full" else preset
    return dict(account=account, dataset=dataset, strategy=strategy, seed=seed,
                preset=preset, prefix=prefix, **extra)


def e1(prefix):
    """E1 closed-end baselines, split so both accounts carry similar GPU time."""
    return (
        [run("towhaelahi", "rad", "Explanation", s, prefix=prefix) for s in (42, 43, 44)]
        + [run("towhaelahi", "slake", "without_R", s, prefix=prefix) for s in (42, 43, 44)]
        + [run("towhaelahi", "rad", "Reasoning", 42, prefix=prefix)]
        + [run("codegiveway", "slake", "Explanation", s, prefix=prefix) for s in (42, 43, 44)]
        + [run("codegiveway", "rad", "without_R", s, prefix=prefix) for s in (42, 43, 44)]
        + [run("codegiveway", "slake", "Reasoning", 42, prefix=prefix)]
    )


# Slug prefixes: "mtbn-v2"  = v2 with patience 5/5/3, min_epochs 25% (superseded,
#                              undertrained -- docs/10). Kept so its results stay fetchable.
#                 "mtbn-v2b" = v2 with patience 15/15/5, min_epochs 50% (current).
PLANS = {
    "sanity": [run("towhaelahi", "rad", "Explanation", 42, preset="sanity")],
    # E5 English control (same pipeline, UnifiedQA-T5-base, English JSONs, same splits/features).
    "ensanity": [run("codegiveway", "rad", "Explanation", 42, preset="sanity", lang="en")],
    "e5": [   # spread over the slots free at 14:40 so all 6 start at once
        run("towhaelahi", "rad", "without_R", 42, lang="en"),
        run("towhaelahi", "rad", "Explanation", 42, lang="en"),
        run("codegiveway", "rad", "Explanation", 42, lang="en", ans_weight=0.5),
        run("codegiveway", "slake", "without_R", 42, lang="en"),
        run("sefatahmedakkhor", "slake", "Explanation", 42, lang="en"),
        run("nishatibnatmaimuna", "slake", "Explanation", 42, lang="en", ans_weight=0.5),
    ],
    # E5b (2026-11-24): seeds 43/44 of the 4 E5 conditions -> 3-seed EN vs BN delta table.
    "e5b": [
        run("mdsafiulalamsarker", "rad", "without_R", 43, lang="en"),
        run("mdsafiulalamsarker", "rad", "without_R", 44, lang="en"),
        run("mdsafiulalamsarker", "rad", "Explanation", 43, lang="en"),
        run("towhaelahi", "rad", "Explanation", 44, lang="en"),
        run("towhaelahi", "rad", "Explanation", 43, lang="en", ans_weight=0.5),
        run("codegiveway", "rad", "Explanation", 44, lang="en", ans_weight=0.5),
        run("codegiveway", "slake", "without_R", 43, lang="en"),
        run("sefatahmedakkhor", "slake", "without_R", 44, lang="en"),
        run("sefatahmedakkhor", "slake", "Explanation", 43, lang="en"),
        run("nishatibnatmaimuna", "slake", "Explanation", 44, lang="en"),
        run("towhaelahi", "slake", "Explanation", 43, lang="en", ans_weight=0.5),  # moved from nishat 11-24
        run("mdsafiulalamsarker", "slake", "Explanation", 44, lang="en", ans_weight=0.5),
    ],
    # Path + BiomedCLIP (2026-11-24): DETR Path ignores the image (zero = real). s42 first on the
    # free mdsafiulalamsarker slot; seeds 43/44 to follow if it helps.
    "pathbmc": [
        run("mdsafiulalamsarker", "path", "Explanation", 42, ans_weight=0.5, features="biomedclip"),
        run("mdsafiulalamsarker", "path", "without_R", 42, features="biomedclip"),
    ],
    # Vanilla Explanation + BMC, seed 43 (s42: RAD 64.0, SLAKE 68.8 vs AARL-full 75.2 / 87.6).
    "e3d": [
        run("codegiveway", "slake", "Explanation", 43, features="biomedclip"),
        run("codegiveway", "rad", "Explanation", 43, features="biomedclip"),
    ],
    # Path: pre-flight sanity, then E1 baselines + alpha=0.5 over 3 seeds (9 runs).
    "pathsanity": [run("towhaelahi", "path", "Explanation", 42, preset="sanity")],
    "pathe1": [   # queued behind e5/e3b; 2 per account except nishatibnatmaimuna
        run("towhaelahi", "path", "without_R", 42),
        run("codegiveway", "path", "without_R", 43),
        run("sefatahmedakkhor", "path", "without_R", 44),
        run("mdsafiulalamsarker", "path", "Explanation", 42),
        run("towhaelahi", "path", "Explanation", 43),
        run("codegiveway", "path", "Explanation", 44),
        run("mdsafiulalamsarker", "path", "Explanation", 42, ans_weight=0.5),
        run("sefatahmedakkhor", "path", "Explanation", 43, ans_weight=0.5),
        run("nishatibnatmaimuna", "path", "Explanation", 44, ans_weight=0.5),
    ],
    # E3c: AARL-full = alpha 0.5 + BiomedCLIP (no warm-up: E3b showed warm-up does not help).
    # 3 seeds (s42 exists from e3b) + w/o-R + BMC x3 + vanilla Expl + BMC s42, on RAD and SLAKE.
    "e3c": [
        run(acc, ds, strat, seed, features="biomedclip", **kw)
        for (acc, ds, strat, seed, kw) in [
            ("nishatibnatmaimuna", "slake", "Explanation", 43, {"ans_weight": 0.5}),
            ("sefatahmedakkhor", "slake", "Explanation", 44, {"ans_weight": 0.5}),
            ("mdsafiulalamsarker", "rad", "Explanation", 43, {"ans_weight": 0.5}),
            ("towhaelahi", "rad", "Explanation", 44, {"ans_weight": 0.5}),
            ("codegiveway", "slake", "without_R", 42, {}),
            ("nishatibnatmaimuna", "slake", "without_R", 43, {}),
            ("sefatahmedakkhor", "slake", "without_R", 44, {}),
            ("mdsafiulalamsarker", "rad", "without_R", 42, {}),
            ("towhaelahi", "rad", "without_R", 43, {}),
            ("codegiveway", "rad", "without_R", 44, {}),
            ("nishatibnatmaimuna", "slake", "Explanation", 42, {}),
            ("sefatahmedakkhor", "rad", "Explanation", 42, {}),
        ]
    ],
    # E3b: {DETR, BiomedCLIP} x {plain, gate bias -1 + vision warm-up} at alpha=0.5,
    # Explanation, seed 42. The DETR/plain cell is the existing e2b/week2 aw50 run.
    "e3b": [
        run(acc, ds, "Explanation", 42, ans_weight=0.5, **sw)
        for ds, accs in (("rad", ("sefatahmedakkhor", "mdsafiulalamsarker", "nishatibnatmaimuna")),
                         ("slake", ("mdsafiulalamsarker", "nishatibnatmaimuna", "sefatahmedakkhor")))
        for acc, sw in zip(accs, (dict(features="biomedclip"),
                                  dict(gate_bias=-1.0, vision_warmup=1),
                                  dict(features="biomedclip", gate_bias=-1.0, vision_warmup=1)))
    ],
    # E3 pre-flight: BiomedCLIP features alone, and with all AARL visual switches.
    # Pushed from co-author accounts so they also prove the dataset share works.
    "bmcsanity": [
        run("mdsafiulalamsarker", "rad", "Explanation", 42, preset="sanity", features="biomedclip"),
        run("sefatahmedakkhor", "rad", "Explanation", 42, preset="sanity", features="biomedclip",
            gate_bias=-1.0, vision_warmup=1),
    ],
    "week1": e1("mtbn-v2"),
    # v2b E1 + E2 over 5 accounts (2026-11-21), balanced by estimated GPU-h
    # (T4: ~0.033 h/epoch with rationale, ~0.011 without; caps 40 RAD / 60 SLAKE).
    # The first four runs were already running on towhaelahi/codegiveway when the
    # other accounts joined, so they stay there.
    "e1": (
        [run("towhaelahi", "rad", "Explanation", s) for s in (42, 43, 44)]          # ~3.9 h
        + [run("codegiveway", "slake", "Explanation", s) for s in (42, 43, 44)]     # ~6.0 h
        + [run("sefatahmedakkhor", "rad", "Reasoning", 42),
           run("sefatahmedakkhor", "slake", "Reasoning", 42)]                       # ~4.7 h
        + [run("sefatahmedakkhor", "rad", "without_R", s) for s in (42, 43, 44)]
        + [run("mdsafiulalamsarker", "slake", "without_R", s) for s in (42, 43, 44)]
    ),
    # E2 (docs/05): segment-balanced loss sweep, seed 42, Explanation.
    "week2": (
        [run("mdsafiulalamsarker", "rad", "Explanation", 42, ans_weight=a)
         for a in (0.5, 0.8, 0.95)]                                                 # ~6.0 h incl. slake w/o-R
        + [run("nishatibnatmaimuna", "slake", "Explanation", 42, ans_weight=a)
           for a in (0.5, 0.8, 0.95)]                                               # ~6.0 h
    ),
    # E2b (doc 11): alpha=0.5 over 3 seeds on both datasets (s42 is in week2),
    # plus alpha=0.3 to check below 0.5. Placed on accounts with free slots.
    "e2b": (
        [run("towhaelahi", "rad", "Explanation", s, ans_weight=0.5) for s in (43, 44)]   # ~2.7 h
        + [run("codegiveway", "slake", "Explanation", 44, ans_weight=0.5)]                # ~2.0 h
        + [run("nishatibnatmaimuna", "slake", "Explanation", 43, ans_weight=0.5),
           run("nishatibnatmaimuna", "slake", "Explanation", 42, ans_weight=0.3)]         # ~4.0 h
        + [run("sefatahmedakkhor", "rad", "Explanation", 42, ans_weight=0.3)]             # ~1.4 h
    ),
}


def slug_for(r: dict) -> str:
    tag = "sanity" if r["preset"] == "sanity" else STRAT_TAG[r["strategy"]]
    slug = f"{r.get('prefix', 'mtbn-v2b')}-{r['dataset']}-closed-{tag}-s{r['seed']}"
    if r.get("ans_weight") is not None:
        slug += f"-aw{round(r['ans_weight'] * 100):02d}"
    if r.get("gate_bias") is not None:
        slug += f"-gb{abs(r['gate_bias']):g}".replace(".", "p")
    if r.get("vision_warmup"):
        slug += f"-vw{r['vision_warmup']}"
    if r.get("features", "detr") != "detr":
        slug += f"-{FEATURE_TAG[r['features']]}"
    if r.get("lang", "bn") != "bn":
        slug += f"-{r['lang']}"
    if r.get("lr") is not None:
        slug += "-lr" + f"{r['lr']:.0e}".replace("e-0", "e")
    return slug


def render_notebook(r: dict) -> Path:
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "PRESET   =" in "".join(c["source"]))
    src = "".join(cell["source"])
    sets = {
        "PRESET  ": f'"{r["preset"]}"',
        "DATASET ": f'"{r["dataset"]}"',
        "STRATEGY": f'"{r["strategy"]}"',
        "SEED    ": str(r["seed"]),
        "PROTOCOL": '"v2"',
    }
    if r.get("ans_weight") is not None:
        sets["ANS_WEIGHT    "] = repr(float(r["ans_weight"]))
    if r.get("gate_bias") is not None:
        sets["GATE_BIAS     "] = repr(float(r["gate_bias"]))
    if r.get("vision_warmup"):
        sets["VISION_WARMUP "] = str(int(r["vision_warmup"]))
    if r.get("features", "detr") != "detr":
        sets["FEATURES      "] = f'"{r["features"]}"'
    if r.get("lang", "bn") != "bn":
        sets["LANG          "] = f'"{r["lang"]}"'
        sets["MODEL         "] = f'"{r.get("model") or LANG_MODEL[r["lang"]]}"'
    if r.get("lr") is not None:
        sets["LR            "] = repr(float(r["lr"]))
    for name, value in sets.items():
        src, n = re.subn(rf"^{name} = \S+", f"{name} = {value}", src, count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"config line {name.strip()} not found in {TEMPLATE.name}")
    cell["source"] = src.splitlines(keepends=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug_for(r)}.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def push(runner: KaggleRunner, r: dict):
    nb = render_notebook(r)
    slug = slug_for(r)
    job = runner.submit_job(RunOptions(
        notebook=str(nb), account_id=r["account"], slug=slug,
        title=slug.replace("-", " "),             # title must slugify back to the id
        accelerator="NvidiaTeslaT4", enable_gpu=True, enable_internet=True,
        dataset_sources=DATASETS
        + ([FEATURE_DATASETS[r["features"]]] if r.get("features", "detr") != "detr" else [])
        + ([LANG_DATASET[r["lang"]]] if r.get("lang", "bn") != "bn" else []),
    ))
    return runner.push_job(job.id)


def run_state(runner: KaggleRunner, slug: str, refresh: bool) -> str:
    """pending | active | done | failed, from the newest job record for `slug`."""
    jobs = [j for j in runner.list_jobs(limit=10_000) if j.kernel_slug == slug]
    if not jobs:
        return "pending"
    job = runner.check_status(jobs[0].id) if refresh else jobs[0]
    if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
        return "active"
    if job.status in (JobStatus.COMPLETE, JobStatus.DOWNLOADED):
        return "done"
    if job.status == JobStatus.CANCELLED:
        return "failed"
    if job.status == JobStatus.ERROR:
        # A run that executed and crashed reports KernelWorkerStatus.ERROR.
        # Anything else (rejected push, kernel never created -> 404) never ran.
        text = f"{job.last_log_snippet or ''} {job.error_message or ''}"
        return "failed" if "KernelWorkerStatus.ERROR" in text else "pending"
    return "pending"


def schedule(runner: KaggleRunner, plan: list, limit: int, poll: int) -> int:
    grace: dict = {}                 # slug -> push time; Kaggle needs a moment to create it
    reported: dict = {}
    while True:
        states = {}
        for r in plan:
            slug = slug_for(r)
            fresh = time.time() - grace.get(slug, 0) < 240
            states[slug] = "active" if fresh else run_state(runner, slug, refresh=True)
            if states[slug] != reported.get(slug):
                print(time.strftime("%H:%M:%S"), f"{r['account']:12s} {slug:34s} -> {states[slug]}",
                      flush=True)
                reported[slug] = states[slug]

        for acct in sorted({r["account"] for r in plan}):
            busy = sum(states[slug_for(r)] == "active" for r in plan if r["account"] == acct)
            for r in plan:
                if busy >= limit:
                    break
                slug = slug_for(r)
                if r["account"] != acct or states[slug] != "pending":
                    continue
                job = push(runner, r)
                if job.status == JobStatus.ERROR:
                    print(time.strftime("%H:%M:%S"), f"   push rejected {slug}: "
                          f"{(job.error_message or '').strip()[:160]}", flush=True)
                    break                    # account is full after all; retry next poll
                print(time.strftime("%H:%M:%S"), f"   pushed {acct}/{slug}", flush=True)
                grace[slug] = time.time()
                states[slug] = reported[slug] = "active"
                busy += 1

        if all(states[slug_for(r)] in ("done", "failed") for r in plan):
            failed = [s for s, st in states.items() if st == "failed"]
            print(time.strftime("%H:%M:%S"), f"ALL FINISHED: {len(plan) - len(failed)} done, "
                  f"{len(failed)} failed {failed}", flush=True)
            return 1 if failed else 0
        # Sleep in short chunks against the wall clock: a single long sleep is
        # paused by a host suspend, which once stalled polling for ~5 h.
        wake = time.time() + poll
        while time.time() < wake:
            time.sleep(min(30, max(0.0, wake - time.time())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True,
                    help=f"One plan or several joined by commas: {', '.join(PLANS)}")
    ap.add_argument("--dry-run", action="store_true", help="Render notebooks, push nothing")
    ap.add_argument("--only", help="Push only runs whose slug contains this text")
    ap.add_argument("--schedule", action="store_true",
                    help="Keep each account at --max-concurrent runs until the plan is done")
    ap.add_argument("--max-concurrent", type=int, default=2, help="Kaggle's GPU session limit")
    ap.add_argument("--poll", type=int, default=300, help="Seconds between status checks")
    args = ap.parse_args()

    runner = KaggleRunner()
    unknown = [p for p in args.plan.split(",") if p not in PLANS]
    if unknown:
        ap.error(f"unknown plan(s) {unknown}; choose from {sorted(PLANS)}")
    plan = [r for p in args.plan.split(",") for r in PLANS[p]
            if not args.only or args.only in slug_for(r)]
    for r in plan:
        if not runner.account_manager.get_account(r["account"]):
            print(f"!! account {r['account']} is not configured in KaggleFleet (kfleet account list)")
            return 1

    if args.schedule and not args.dry_run:
        return schedule(runner, plan, args.max_concurrent, args.poll)

    failed = 0
    for r in plan:
        slug = slug_for(r)
        line = f"{r['account']:12s} {slug:34s} preset={r['preset']}"
        if args.dry_run:
            print("[dry-run]", line, "->", render_notebook(r).relative_to(ROOT))
            continue
        job = push(runner, r)
        ok = job.status != JobStatus.ERROR
        failed += not ok
        print(("[pushed] " if ok else "[FAILED] ") + line, "|", job.kernel_id)
        if not ok:
            print("         ", (job.error_message or "").strip()[:400])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
