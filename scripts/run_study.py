"""
Orchestrator for independent replications with RANDOM seeds.

For each replication i:
  1) draw random seeds (sample_seeds.py),
  2) collect by snowball (collect.py),
  3) densify the induced subgraph concurrently (densify.py),
  4) analyse (analyze.py),
each in its own directory runs/run_i/. The key metrics are then aggregated across
replications (mean +/- standard deviation), so that what is measured is the STABILITY
of the findings rather than a single point estimate.

Usage:
  ./venv/bin/python run_study.py --runs 5 --n 3000 --k 1000 --seeds-per-run 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

SCRIPTS = Path(__file__).resolve().parent      # the scripts/ directory, to launch its siblings
ROOT = SCRIPTS.parent                           # repository root (for runs/)
PY = sys.executable


def run(cmd: list[str], env: dict | None = None, capture: bool = False) -> str:
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, text=True,
                       capture_output=capture, check=True)
    return r.stdout if capture else ""


def get(d: dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# Key metrics tracked across replications (label -> path inside report.json).
METRICS = {
    "gini_posts (production)": "RQ1_production.gini_posts",
    "gini_in_degree (attention)": "RQ2_attention.gini_in_degree",
    "gini_followers (global)": "RQ2b_follower_distribution.gini_followers",
    "spearman_posts_indeg": "RQ3_overlap.spearman_posts_vs_indegree.rho",
    "attention_top1%": "RQ4_concentration.attention_top_1pct",
    "gini_pagerank (power)": "RQ_power.gini_pagerank",
    "power_top1%": "RQ_power.pagerank_top_1pct_share",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--seeds-per-run", type=int, default=10)
    ap.add_argument("--scan", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    base = ROOT / args.out
    base.mkdir(exist_ok=True)
    reports = []

    for i in range(args.runs):
        rdir = base / f"run_{i}"
        ddir, odir = rdir / "data", rdir / "results"
        ddir.mkdir(parents=True, exist_ok=True)
        odir.mkdir(parents=True, exist_ok=True)
        env = {"RRSS_DATA_DIR": str(ddir), "RRSS_OUT_DIR": str(odir)}
        print(f"\n===== REPLICATION {i+1}/{args.runs} =====", flush=True)

        seeds = run([PY, str(SCRIPTS / "sample_seeds.py"), "--n", str(args.seeds_per_run),
                     "--scan", str(args.scan), "--seed", str(1000 + i)],
                    capture=True).split()
        if not seeds:
            print("  (no seeds, skipping)"); continue
        print(f"  seeds: {seeds}", flush=True)

        run([PY, str(SCRIPTS / "collect.py"), "--seeds", *seeds, "--n", str(args.n),
             "--depth", str(args.depth), "--k", str(args.k)], env=env)
        run([PY, str(SCRIPTS / "densify.py"), "--k", str(args.k), "--workers", str(args.workers)], env=env)
        run([PY, str(SCRIPTS / "analyze.py")], env=env)

        rep = json.loads((odir / "report.json").read_text())
        rep["_seeds"] = seeds
        reports.append(rep)

    # --- Aggregation across replications ---
    agg = {"n_runs": len(reports), "params": vars(args), "metrics": {}}
    print("\n" + "=" * 64)
    print(f"AGGREGATE over {len(reports)} replications (mean +/- SD)")
    print("=" * 64)
    for label, path in METRICS.items():
        vals = [v for r in reports if (v := get(r, path)) is not None]
        if not vals:
            continue
        m, sd = mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)
        agg["metrics"][label] = {"mean": round(m, 4), "sd": round(sd, 4),
                                 "values": [round(v, 4) for v in vals]}
        print(f"  {label:32s} {m:7.4f} ± {sd:.4f}")

    (base / "aggregate.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n-> {base / 'aggregate.json'}")


if __name__ == "__main__":
    main()
