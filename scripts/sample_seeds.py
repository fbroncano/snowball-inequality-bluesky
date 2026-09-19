"""
Random seed selection, to avoid the bias of institutional seed accounts.

Draws uniformly from the full relay census written by uniform_sample.py when it is
available. Failing that it falls back to a scan of the relay listing
(com.atproto.sync.listRepos), which enumerates every repository (account) on the
protocol; reservoir sampling over several pages gives a near-uniform set of DIDs.
That fallback reads from the start of a sequence ordered by creation date and therefore
over-samples the platform's earlier accounts.

Either way the candidates are filtered to active accounts (handle resolvable, at least
one post) so that they are usable as snowball seeds.

Usage:
  ./venv/bin/python sample_seeds.py --n 20 --scan 5000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import requests

from collect import BlueskyClient

RELAY = "https://bsky.network/xrpc/com.atproto.sync.listRepos"


def scan_dids(scan: int) -> list[str]:
    """Walks relay pages and returns up to `scan` DIDs of active repositories."""
    s = requests.Session()
    dids: list[str] = []
    cursor = None
    while len(dids) < scan:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = s.get(RELAY, params=params, timeout=30)
        if r.status_code != 200:
            break
        data = r.json()
        for repo in data.get("repos", []):
            if repo.get("active", True):
                dids.append(repo["did"])
        cursor = data.get("cursor")
        if not cursor:
            break
    return dids


CENSUS = Path(__file__).resolve().parent.parent / "data" / "relay_dids.txt"


def sample_census(n_pool: int) -> list[str]:
    """UNIFORM sampling over the full census enumerated by uniform_sample.py.

    Preferable to scan_dids(): that function reads the relay listing from the start,
    and since the relay orders by creation sequence it returns the OLDEST accounts
    rather than a uniform sample of the platform.
    """
    total = sum(1 for _ in CENSUS.open())
    idx = set(random.sample(range(total), min(n_pool, total)))
    out = []
    with CENSUS.open() as fh:
        for i, line in enumerate(fh):
            if i in idx:
                out.append(line.split("\t")[0])
    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="Number of random seeds to return")
    ap.add_argument("--scan", type=int, default=5000, help="Size of the relay scan")
    ap.add_argument("--seed", type=int, default=None, help="Semilla RNG (reproducibilidad)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Prefer the full census when it exists and sample uniformly from it. Otherwise
    # fall back to the relay scan, which is biased towards the oldest accounts.
    if CENSUS.exists():
        pool = sample_census(args.scan)
    else:
        print("[warning] no census at data/relay_dids.txt; falling back to the relay "
              "scan, which is BIASED towards older accounts. Run this first:\n"
              "  uniform_sample.py --enumerate", flush=True)
        pool = scan_dids(args.scan)
        random.shuffle(pool)

    # Keep accounts with a resolvable profile and >=1 post (usable snowball seeds).
    client = BlueskyClient()
    seeds: list[str] = []
    for did in pool:
        if len(seeds) >= args.n:
            break
        prof = client.get_profile(did)
        if prof and (prof.get("postsCount") or 0) > 0 and prof.get("handle"):
            seeds.append(prof["handle"])

    print(" ".join(seeds))


if __name__ == "__main__":
    main()
