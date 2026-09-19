"""
NEAR-UNBIASED estimation of follower concentration on Bluesky.

Unlike the snowball, no part of the graph is reconstructed here. The relay's repository
census is enumerated (com.atproto.sync.listRepos), accounts are drawn UNIFORMLY from
that census, and the Gini is applied to their follower counters. The counter is global,
reported by the platform, so it does not depend on which other accounts happen to be in
the sample. This is the reference estimand against which the in-degree Gini of the
induced subgraph is set.

Two phases, the first of which is cached:
  1) --enumerate : walks the relay and writes every DID to data/relay_dids.txt.
                   The relay cursor is an integer sequence number rather than an opaque
                   token, so the range can be partitioned and traversed in parallel, and
                   the walk is resumable from a per-worker cursor file.
  2) sampling    : draws uniformly from that census, hydrates the profiles with
                   getProfiles, and reports the Gini of the follower counts with
                   percentile bootstrap intervals.

Output: results/uniform_sample.json, results/uniform_big.json and the raw counter arrays
in results/uniform_*_raw.npz.

Usage:
  ./venv/bin/python uniform_sample.py --enumerate --workers 16
  ./venv/bin/python uniform_sample.py --big --n 100000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import threading

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from collect import BlueskyClient


def hydrate_parallel(dids, client, workers=16):
    """getProfiles in batches of 25, spread across threads. Returns profiles."""
    batches = [dids[i:i + 25] for i in range(0, len(dids), 25)]
    out = []
    lock = threading.Lock()

    def fetch(batch):
        profs = client.get_profiles(batch)
        with lock:
            out.extend(profs)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, batches))
    return out


def boot_ci(x, stat, B=2000, alpha=0.05, rng=None):
    """Percentile confidence interval by non-parametric bootstrap."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = stat(x[rng.integers(0, n, n)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results"
RELAY = "https://bsky.network/xrpc/com.atproto.sync.listRepos"
DIDS_FILE = DATA / "relay_dids.txt"
CURSOR_FILE = DATA / "relay_cursor.json"


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def top_share(x, frac=0.01) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(len(x) * frac))
    return float(x[:k].sum() / x.sum())


# --------------------------------------------------------------------------- phase 1
def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "research-pilot/0.1 (academic study)"
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))
    return s


def _page(s, cursor, limit=1000):
    params = {"limit": limit}
    if cursor is not None:
        params["cursor"] = str(cursor)
    for attempt in range(5):
        try:
            r = s.get(RELAY, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", 2 ** (attempt + 1)))); continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt); continue
        return {}
    return {}


def probe_max_cursor() -> int:
    """The relay cursor is an integer sequence number. Locates its upper end."""
    s = _session()
    lo, hi = 0, 1000
    while hi < 10 ** 11:
        d = _page(s, hi, 1)
        if not d.get("repos"):
            break
        lo, hi = hi, hi * 4
        time.sleep(0.05)
    a, b = lo, hi
    for _ in range(40):
        if b - a <= 1:
            break
        m = (a + b) // 2
        if _page(s, m, 1).get("repos"):
            a = m
        else:
            b = m
        time.sleep(0.03)
    return a


def _worker(idx: int, start: int, stop: int, pause: float) -> tuple[int, int]:
    """Enumerates [start, stop) into its own file. Resumable."""
    part = DATA / f"relay_dids.part{idx:03d}"
    ck = DATA / f"relay_dids.part{idx:03d}.cursor"
    cursor = start
    if ck.exists():
        cursor = int(ck.read_text().strip() or start)
    s = _session()
    n = 0
    with part.open("a") as fh:
        while cursor is not None and cursor < stop:
            d = _page(s, cursor, 1000)
            repos = d.get("repos", [])
            if not repos:
                break
            for repo in repos:
                fh.write(f"{repo['did']}\t{'1' if repo.get('active', True) else '0'}\n")
            n += len(repos)
            nxt = d.get("cursor")
            if nxt is None:
                break
            cursor = int(nxt)
            ck.write_text(str(cursor))
            fh.flush()
            time.sleep(pause)
    return idx, n


def enumerate_relay(workers: int, pause: float, max_cursor: int | None) -> int:
    """Enumerates the full census, splitting the cursor range across `workers` threads."""
    DATA.mkdir(parents=True, exist_ok=True)
    if max_cursor is None:
        print("[probe] locating the end of the relay sequence...", flush=True)
        max_cursor = probe_max_cursor()
    print(f"[probe] maximum cursor ~ {max_cursor:,}", flush=True)

    edges = [max_cursor * i // workers for i in range(workers + 1)]
    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, i, edges[i], edges[i + 1], pause): i for i in range(workers)}
        done = 0
        for f in as_completed(futs):
            idx, n = f.result()
            total += n; done += 1
            print(f"  worker {idx:3d} finished with {n:,} DIDs "
                  f"({done}/{workers} threads, {time.time()-t0:.0f}s)", flush=True)

    # Merge and deduplicate. The ranges overlap at their boundaries, but the census runs
    # to tens of millions of lines, so deduplication uses `sort -u` (external sort,
    # bounded in memory) rather than a dict held in RAM.
    parts = [p for p in sorted(DATA.glob("relay_dids.part*")) if p.suffix != ".cursor"]
    if not parts:
        raise SystemExit("No se genero ninguna particion")
    tmp = DATA / "relay_dids.merging"
    env = {**os.environ, "LC_ALL": "C", "TMPDIR": str(DATA)}
    with tmp.open("w") as fh:
        cat = subprocess.Popen(["cat", *[str(p) for p in parts]], stdout=subprocess.PIPE)
        subprocess.run(["sort", "-u", "-S", "512M"], stdin=cat.stdout, stdout=fh,
                       env=env, check=True)
        cat.stdout.close(); cat.wait()
    tmp.replace(DIDS_FILE)
    n_unique = int(subprocess.run(["wc", "-l", str(DIDS_FILE)], capture_output=True,
                                  text=True, check=True).stdout.split()[0])
    size_mb = DIDS_FILE.stat().st_size / 1e6
    print(f"[enumerate] {n_unique:,} unique DIDs ({total:,} read) "
          f"en {time.time()-t0:.0f}s -> {DIDS_FILE} ({size_mb:,.0f} MB)")
    return n_unique


# --------------------------------------------------------------------------- phase 2
def load_census(active_only: bool) -> list[str]:
    if not DIDS_FILE.exists():
        raise SystemExit("No census found. Run --enumerate first.")
    dids = []
    for line in DIDS_FILE.open():
        parts = line.rstrip("\n").split("\t")
        did = parts[0]
        active = (len(parts) < 2) or parts[1] == "1"
        if active or not active_only:
            dids.append(did)
    return dids


def sample_once(census: list[str], n: int, client: BlueskyClient, rng: random.Random) -> dict:
    picked = rng.sample(census, min(n, len(census)))
    profiles = hydrate_parallel(picked, client)
    fc = np.array([p.get("followersCount", 0) or 0 for p in profiles], dtype=float)
    pc = np.array([p.get("postsCount", 0) or 0 for p in profiles], dtype=float)
    resolved = len(profiles)
    # Two defensible populations: ALL resolved accounts, or only those that have posted
    # at least once. Both are reported, because the choice moves the Gini.
    posted = fc[pc > 0]
    # Persist the raw counters, so any statistic can be recomputed later without
    # re-hydrating tens of thousands of profiles.
    raw = {"followers": fc.tolist(), "posts": pc.tolist()}
    return {
        "_raw": raw,
        "requested": len(picked),
        "resolved": resolved,
        "resolution_rate": round(resolved / max(1, len(picked)), 4),
        "gini_followers_all": round(gini(fc), 4),
        "gini_followers_active": round(gini(posted), 4) if len(posted) else None,
        "n_active": int(len(posted)),
        "top1_share_all": round(top_share(fc), 4) if resolved else None,
        "top1_share_active": round(top_share(posted), 4) if len(posted) else None,
        "median_followers_active": float(np.median(posted)) if len(posted) else None,
        "mean_followers_active": round(float(np.mean(posted)), 1) if len(posted) else None,
        "gini_posts": round(gini(pc), 4),
        "median_followers": float(np.median(fc)) if resolved else None,
        "mean_followers": round(float(np.mean(fc)), 1) if resolved else None,
        "max_followers": float(np.max(fc)) if resolved else None,
        "pct_zero_followers": round(float((fc == 0).mean()), 4) if resolved else None,
        "pct_under_1000": round(float((fc < 1000).mean()), 4) if resolved else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true", help="Phase 1: walk the relay")
    ap.add_argument("--workers", type=int, default=16, help="Enumeration threads")
    ap.add_argument("--max-cursor", type=int, default=None, help="Skip probing for the end of the sequence")
    ap.add_argument("--pause", type=float, default=0.05)
    ap.add_argument("--sample", action="store_true", help="Phase 2: sample and estimate")
    ap.add_argument("--big", action="store_true", help="One large sample, with bootstrap intervals and a convergence curve")
    ap.add_argument("--n", type=int, default=5000, help="Accounts per sample")
    ap.add_argument("--reps", type=int, default=5, help="Muestras independientes")
    ap.add_argument("--active-only", action="store_true",
                    help="Exclude from the census the repositories marked inactive")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.enumerate:
        enumerate_relay(args.workers, args.pause, args.max_cursor)

    if args.big:
        census = load_census(args.active_only)
        rng = random.Random(args.seed)
        client = BlueskyClient()
        print(f"[census] {len(census):,} accounts. Sampling {args.n:,}...", flush=True)
        t0 = time.time()
        picked = rng.sample(census, min(args.n, len(census)))
        profiles = hydrate_parallel(picked, client, workers=args.workers)
        fc = np.array([p.get("followersCount", 0) or 0 for p in profiles], dtype=float)
        pc = np.array([p.get("postsCount", 0) or 0 for p in profiles], dtype=float)
        print(f"  resolved {len(profiles):,}/{len(picked):,} in {time.time()-t0:.0f}s", flush=True)

        nprng = np.random.default_rng(args.seed)
        out = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "census_size": len(census), "requested": len(picked), "resolved": len(profiles),
            "resolution_rate": round(len(profiles) / len(picked), 4),
            "gini_followers_all": round(gini(fc), 4),
            "gini_followers_all_ci95": boot_ci(fc, gini, rng=nprng),
            "top1_share_all": round(top_share(fc), 4),
            "top1_share_all_ci95": boot_ci(fc, top_share, rng=nprng),
            "gini_posts": round(gini(pc), 4),
            "gini_posts_ci95": boot_ci(pc, gini, rng=nprng),
            "median_followers": float(np.median(fc)),
            "max_followers": float(np.max(fc)),
            "pct_zero_followers": round(float((fc == 0).mean()), 4),
        }
        posted = fc[pc > 0]
        out["gini_followers_active"] = round(gini(posted), 4)
        out["gini_followers_active_ci95"] = boot_ci(posted, gini, rng=nprng)
        out["n_active"] = int(len(posted))

        # Convergence curve: Gini over nested subsamples.
        conv = []
        for m in (1000, 3000, 10000, 30000, 100000, 300000):
            if m > len(fc):
                break
            reps = [gini(nprng.choice(fc, m, replace=False)) for _ in range(20)]
            conv.append({"n": m, "gini_mean": round(float(np.mean(reps)), 4),
                         "gini_sd": round(float(np.std(reps, ddof=1)), 4)})
            print(f"  convergence n={m:>7,}: {conv[-1]['gini_mean']} +/- {conv[-1]['gini_sd']}", flush=True)
        out["convergence"] = conv

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "uniform_big.json").write_text(json.dumps(out, indent=2))
        np.savez_compressed(OUT / "uniform_big_raw.npz", followers=fc, posts=pc)
        print(f"\n[saved] {OUT / 'uniform_big.json'}")
        print(f"  Gini (all)      {out['gini_followers_all']}  CI95 {out['gini_followers_all_ci95']}")
        print(f"  Gini (>=1 post) {out['gini_followers_active']}  CI95 {out['gini_followers_active_ci95']}")
        print(f"  top-1%          {out['top1_share_all']}  CI95 {out['top1_share_all_ci95']}")

    if args.sample:
        census = load_census(args.active_only)
        print(f"[census] {len(census):,} accounts (active_only={args.active_only})")
        rng = random.Random(args.seed)
        client = BlueskyClient()
        reps = []
        for i in range(args.reps):
            t0 = time.time()
            rep = sample_once(census, args.n, client, rng)
            rep["rep"] = i
            reps.append(rep)
            print(f"  rep {i}: Gini(all)={rep['gini_followers_all']} "
                  f"Gini(with posts)={rep['gini_followers_active']} "
                  f"resolved={rep['resolved']}/{rep['requested']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        def agg(key):
            v = [r[key] for r in reps if r.get(key) is not None]
            return {"mean": round(float(np.mean(v)), 4),
                    "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
                    "values": v}

        out = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "census_size": len(census),
            "active_only": args.active_only,
            "n_per_sample": args.n,
            "n_reps": args.reps,
            "gini_followers_all": agg("gini_followers_all"),
            "gini_followers_active": agg("gini_followers_active"),
            "gini_posts": agg("gini_posts"),
            "top1_share_all": agg("top1_share_all"),
            "top1_share_active": agg("top1_share_active"),
            "median_followers_active": agg("median_followers_active"),
            "resolution_rate": agg("resolution_rate"),
            "pct_zero_followers": agg("pct_zero_followers"),
            "pct_under_1000": agg("pct_under_1000"),
            "per_rep": [{k: v for k, v in r.items() if k != "_raw"} for r in reps],
        }
        np.savez_compressed(OUT / "uniform_sample_raw.npz",
                            **{f"followers_{r['rep']}": np.array(r["_raw"]["followers"])
                               for r in reps},
                            **{f"posts_{r['rep']}": np.array(r["_raw"]["posts"])
                               for r in reps})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "uniform_sample.json").write_text(json.dumps(out, indent=2))
        print(f"\n[saved] {OUT / 'uniform_sample.json'}")
        print(f"  Follower Gini (all accounts): "
              f"{out['gini_followers_all']['mean']} +/- {out['gini_followers_all']['sd']}")
        print(f"  Follower Gini (>=1 post only): "
              f"{out['gini_followers_active']['mean']} +/- {out['gini_followers_active']['sd']}")


if __name__ == "__main__":
    main()
