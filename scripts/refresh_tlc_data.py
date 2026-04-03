#!/usr/bin/env python3
"""
Download NYC TLC Yellow (or FHVHV) Parquet files with SHA256 checksums.

Keeps a local manifest so re-runs skip unchanged files. Large binaries stay
out of git — place outputs under `taxi data/` (see backend/config.yaml paths).

Usage:
  python scripts/refresh_tlc_data.py --yellow --year 2025 --months 1 2 3
  python scripts/refresh_tlc_data.py --yellow --year 2025 --months 1-6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "surge-pred-tlc-refresh/1.0"})
    with urlopen(req, timeout=600) as resp, dest.open("wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)


def parse_months(spec: list[str]) -> list[int]:
    out: list[int] = []
    for s in spec:
        if "-" in s and s.count("-") == 1:
            a, b = s.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    for m in out:
        if m < 1 or m > 12:
            raise ValueError(f"Invalid month: {m}")
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(description="Download TLC Parquet with checksum manifest.")
    p.add_argument("--yellow", action="store_true", help="Yellow taxi tripdata")
    p.add_argument("--fhvhv", action="store_true", help="FHVHV tripdata")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--months", nargs="+", required=True, help="e.g. 1 2 3 or 1-6")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("taxi data"),
        help="Directory for .parquet files (default: ./taxi data)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("taxi data") / ".tlc_manifest.json",
        help="Checksum manifest path (default: taxi data/.tlc_manifest.json)",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if checksum matches")
    args = p.parse_args()

    modes: list[tuple[str, str]] = []
    if args.yellow:
        modes.append(("yellow_tripdata", "yellow"))
    if args.fhvhv:
        modes.append(("fhvhv_tripdata", "fhvhv"))
    if not modes:
        print("Specify --yellow and/or --fhvhv", file=sys.stderr)
        return 2

    months = parse_months(args.months)

    manifest: dict = {}
    if args.manifest.is_file():
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
    files_meta = manifest.setdefault("files", {})

    for prefix, _label in modes:
        for m in months:
            name = f"{prefix}_{args.year}-{m:02d}.parquet"
            url = f"{BASE_URL}/{name}"
            dest = args.output_dir / name
            existing = files_meta.get(name, {})
            prev_hash = existing.get("sha256")

            if dest.is_file() and not args.force and prev_hash:
                cur = sha256_file(dest)
                if cur == prev_hash:
                    print(f"OK (cached) {name}")
                    continue

            print(f"Downloading {url} ...")
            try:
                download(url, dest)
            except HTTPError as e:
                print(f"HTTP error for {name}: {e}", file=sys.stderr)
                continue
            except URLError as e:
                print(f"Network error for {name}: {e}", file=sys.stderr)
                return 1

            digest = sha256_file(dest)
            files_meta[name] = {
                "sha256": digest,
                "bytes": dest.stat().st_size,
                "url": url,
            }
            print(f"Wrote {dest} sha256={digest[:16]}...")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest updated: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
