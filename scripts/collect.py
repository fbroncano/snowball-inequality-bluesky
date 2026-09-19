"""
Snowball (BFS) collector for the Bluesky follow graph (AT Protocol).

Reads through the public, unauthenticated interface (public.api.bsky.app). For every
profile it records the platform counters (followers, follows, posts) and the directed
"A follows B" edges whose two endpoints are both inside the sample.

Output (in ./data):
  - nodes.parquet : one profile per row with its attributes.
  - edges.parquet : directed edges (src follows dst), both endpoints in the sample.
  - meta.json     : collection parameters and timestamp, for reproducibility.

Usage:
  ./venv/bin/python collect.py --seeds bsky.app --n 5000 --depth 3 --k 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://public.api.bsky.app/xrpc"
# Data directory overridable by environment, so replications can run in parallel and isolated.
DATA_DIR = Path(os.environ.get("RRSS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


class BlueskyClient:
    """Minimal client over the public API, with retries and rate-limit backoff."""

    def __init__(self, pause: float = 0.02, max_retries: int = 4):
        self.session = requests.Session()
        # Wide connection pool, for concurrent use from threads.
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self.session.mount("https://", adapter)
        self.session.headers["User-Agent"] = "research-pilot/0.1 (academic study)"
        self.pause = pause
        self.max_retries = max_retries

    def _get(self, method: str, params: dict) -> dict:
        url = f"{BASE}/{method}"
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                time.sleep(self.pause)
                return r.json()
            if r.status_code == 429:  # rate limited
                wait = int(r.headers.get("retry-after", 2 ** (attempt + 1)))
                time.sleep(wait)
                continue
            if r.status_code in (400, 404):  # perfil borrado/baneado: no insistir
                return {}
            time.sleep(2 ** attempt)
        return {}

    def get_profile(self, actor: str) -> dict:
        return self._get("app.bsky.actor.getProfile", {"actor": actor})

    def get_profiles(self, actors: list[str]) -> list[dict]:
        """DETAILED profiles (with counters) in batches of up to 25 actors."""
        out: list[dict] = []
        for i in range(0, len(actors), 25):
            batch = actors[i:i + 25]
            data = self._get("app.bsky.actor.getProfiles", {"actors": batch})
            out.extend(data.get("profiles", []))
        return out

    def get_follows(self, actor: str, k: int) -> list[dict]:
        """Returns up to k profiles that `actor` FOLLOWS (out-edges)."""
        return self._paginate("app.bsky.graph.getFollows", actor, "follows", k)

    def _paginate(self, method: str, actor: str, key: str, k: int) -> list[dict]:
        out: list[dict] = []
        cursor = None
        while len(out) < k:
            params = {"actor": actor, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            data = self._get(method, params)
            if not data:
                break
            out.extend(data.get(key, []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return out[:k]


def collect(seeds: list[str], n: int, depth: int, k: int) -> tuple[dict, list[tuple]]:
    """
    BFS along the FOLLOWS relation (I follow -> out-edges).

    Returns:
      nodes: dict did -> attributes
      edges: list of (src_did, dst_did)
    """
    client = BlueskyClient()
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str]] = []
    visited_expanded: set[str] = set()  # nodes whose follows we already fetched

    # Queue of (actor, depth). Seeded with the seed accounts.
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    queued: set[str] = set(seeds)

    def record_profile(p: dict) -> str | None:
        did = p.get("did")
        if not did:
            return None
        if did not in nodes:
            created = p.get("createdAt")
            days_active = None
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    days_active = (datetime.now(timezone.utc) - dt).days
                except ValueError:
                    pass
            nodes[did] = {
                "did": did,
                "handle": p.get("handle"),
                "followers_count": p.get("followersCount"),
                "follows_count": p.get("followsCount"),
                "posts_count": p.get("postsCount"),
                "created_at": created,
                "days_active": days_active,
            }
        return did

    while queue and len(nodes) < n:
        actor, d = queue.popleft()
        prof = client.get_profile(actor)
        if not prof:
            continue
        src_did = record_profile(prof)
        if src_did is None or src_did in visited_expanded:
            continue
        visited_expanded.add(src_did)

        if d >= depth:
            continue  # record the node but do not expand past the depth limit

        follows = client.get_follows(actor, k)
        for f in follows:
            dst_did = record_profile(f)
            if dst_did is None:
                continue
            edges.append((src_did, dst_did))
            handle = f.get("handle")
            if handle and dst_did not in queued and len(nodes) < n * 3:
                queue.append((handle, d + 1))
                queued.add(dst_did)

        if len(nodes) % 100 == 0:
            print(f"  ... {len(nodes)} nodes, {len(edges)} edges", flush=True)

    # --- Hydration: getFollows returns profiles WITHOUT counters; fill them in ---
    # with getProfiles (detailed, batches of 25) for every node missing them.
    pending = [d for d, a in nodes.items() if a.get("followers_count") is None]
    print(f"Hydrating counters for {len(pending)} nodes...", flush=True)
    for i in range(0, len(pending), 25):
        batch = pending[i:i + 25]
        for p in client.get_profiles(batch):
            did = p.get("did")
            if did in nodes:
                created = p.get("createdAt")
                days_active = nodes[did].get("days_active")
                if created and days_active is None:
                    try:
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        days_active = (datetime.now(timezone.utc) - dt).days
                    except ValueError:
                        pass
                nodes[did].update({
                    "followers_count": p.get("followersCount"),
                    "follows_count": p.get("followsCount"),
                    "posts_count": p.get("postsCount"),
                    "created_at": created,
                    "days_active": days_active,
                })
        if (i // 25) % 20 == 0:
            print(f"  ... hydrated {min(i + 25, len(pending))}/{len(pending)}", flush=True)

    return nodes, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="Handles semilla, p.ej. --seeds bsky.app jay.bsky.team")
    ap.add_argument("--n", type=int, default=5000, help="Target number of nodes")
    ap.add_argument("--depth", type=int, default=3, help="Profundidad BFS")
    ap.add_argument("--k", type=int, default=1000,
                    help="Cap on follows fetched per node (sampling of hubs)")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Collection: seeds={args.seeds} n={args.n} depth={args.depth} k={args.k}")
    t0 = time.time()
    nodes, edges = collect(args.seeds, args.n, args.depth, args.k)
    elapsed = time.time() - t0

    # Keep only edges with both endpoints in the sample (induced graph).
    node_ids = set(nodes)
    edges = [(s, d) for s, d in edges if s in node_ids and d in node_ids]

    nodes_df = pd.DataFrame(nodes.values())
    edges_df = pd.DataFrame(edges, columns=["src", "dst"]).drop_duplicates()

    nodes_df.to_parquet(DATA_DIR / "nodes.parquet", index=False)
    edges_df.to_parquet(DATA_DIR / "edges.parquet", index=False)

    meta = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform": "bluesky",
        "method": "snowball_bfs_follows",
        "seeds": args.seeds,
        "n_target": args.n,
        "depth": args.depth,
        "k_per_node": args.k,
        "n_nodes": len(nodes_df),
        "n_edges": len(edges_df),
        "elapsed_seconds": round(elapsed, 1),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  nodes: {len(nodes_df)}")
    print(f"  edges (induced graph): {len(edges_df)}")
    print(f"  -> {DATA_DIR}")


if __name__ == "__main__":
    main()
