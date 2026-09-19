"""
Pseudonymisation filter applied to every artefact before release.

The analysis writes account handles and DIDs into the run metadata (the seed accounts
of each replication) and into the reports (the ranked hub lists). The ethical basis of
the study excludes redistributing individual records, so those identifiers are replaced
here by opaque labels before the JSON files are copied into the public repository.

Each distinct identifier is mapped to `acct_<12 hex>`, the truncated SHA-256 of the
identifier concatenated with a 32-byte random salt. The salt is generated once per
invocation and never written out, so the mapping is one-way and cannot be inverted by
dictionary attack over the public account list. The mapping is consistent within a
single invocation, so an account that appears in two files keeps one label.

Two ranked per-account lists are dropped outright rather than relabelled. A pseudonym
does not hide the account at the head of a follower ranking, since its counter alone
identifies it, and neither list feeds any figure or table in the paper. The aggregate
scalars computed from them, such as the top-percentile shares, are kept.

Every other field is passed through unchanged: the released files therefore retain the
counters and distributions that the figures and tables rest on.

Usage:
  ./venv/bin/python scripts/pseudonymise.py SOURCE_DIR DEST_DIR
"""
import hashlib
import json
import os
import re
import secrets
import shutil
import sys

# A Bluesky handle (foo.bsky.social, example.com) or an AT Protocol DID.
IDENT = re.compile(r"^(did:[a-z]+:[a-z0-9]+|[a-z0-9-]+(\.[a-z0-9-]+)+)$", re.I)
# Keys whose string values are account identifiers.
IDENT_KEYS = {"handle", "did", "seed", "actor", "account"}
# Ranked per-account listings, removed rather than relabelled.
DROP_KEYS = {"top_10_hubs", "top_10_by_power"}

_SALT = secrets.token_bytes(32)
_SEEN = {}


def pseudonym(value):
    if value not in _SEEN:
        digest = hashlib.sha256(_SALT + value.encode("utf-8")).hexdigest()
        _SEEN[value] = "acct_" + digest[:12]
    return _SEEN[value]


def scrub(obj, key=None):
    if isinstance(obj, dict):
        return {k: scrub(v, k) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [scrub(v, key) for v in obj]
    if isinstance(obj, str):
        # Replace when the key says it is an identifier, or when the value is a bare
        # handle sitting in an identifier-bearing list such as `seeds`.
        if key in IDENT_KEYS or (key and key.rstrip("s") in IDENT_KEYS and IDENT.match(obj)):
            return pseudonym(obj)
    return obj


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    n_files = 0
    for root, _, files in os.walk(src):
        for name in sorted(files):
            source = os.path.join(root, name)
            target = os.path.join(dst, os.path.relpath(source, src))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if name.endswith(".json"):
                with open(source, encoding="utf-8") as fh:
                    data = json.load(fh)
                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(scrub(data), fh, ensure_ascii=False, indent=2)
            else:
                shutil.copy2(source, target)
            n_files += 1
    print(f"{n_files} files written to {dst}; {len(_SEEN)} identifiers pseudonymised")


if __name__ == "__main__":
    main()
