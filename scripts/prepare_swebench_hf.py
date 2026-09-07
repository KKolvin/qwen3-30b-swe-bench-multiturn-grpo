#!/usr/bin/env python
"""Download a SWE-bench dataset from Hugging Face and convert it to verl parquet.

Unlike ``prepare_swebench.py`` (which loads a local ``load_from_disk`` copy),
this pulls a *gradable* SWE-bench variant straight from the Hub — one whose
(repo, version) pairs are keyed in the installed ``swebench`` harness so the
binary reward can actually be computed and whose instance images are published
on Docker Hub.

Networking: the datasets/hf_hub file resolver rejects the Xet CDN redirect
behind an HTTP proxy, so we resolve the parquet file list via the Hub API and
download the raw parquet with ``requests`` (which follows the 302 to the CDN
through the proxy). Point ``--proxy`` (or ``HTTPS_PROXY``) at your egress proxy.

Usage:
    python scripts/prepare_swebench_hf.py                       # Verified, test split
    python scripts/prepare_swebench_hf.py --dataset SWE-bench/SWE-bench_Lite
    python scripts/prepare_swebench_hf.py --proxy http://127.0.0.1:12233
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEFAULT_PROXY = "http://127.0.0.1:12233"


def _set_proxy(proxy: str | None) -> str | None:
    proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = proxy
    return proxy


def _download_split_parquets(dataset: str, split: str, proxy: str | None, dest_dir: str) -> list[str]:
    """List parquet files for `split` via the Hub API, download each via requests."""
    import requests
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(dataset, repo_type="dataset")
    # HF stores split shards as data/<split>-NNNNN-of-MMMMM.parquet
    shards = sorted(f for f in files if f.endswith(".parquet") and os.path.basename(f).startswith(f"{split}-"))
    if not shards:
        # Fallback: some repos put a single parquet under data/ without split prefix.
        shards = sorted(f for f in files if f.endswith(".parquet"))
    if not shards:
        raise SystemExit(f"No parquet files found in {dataset} for split {split!r}. Files: {files}")

    os.makedirs(dest_dir, exist_ok=True)
    proxies = {"http": proxy, "https": proxy} if proxy else None
    local = []
    for shard in shards:
        url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{shard}"
        out = os.path.join(dest_dir, os.path.basename(shard))
        print(f"  downloading {shard} ...")
        with requests.get(url, proxies=proxies, allow_redirects=True, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        local.append(out)
    return local


def _load_rows(parquet_paths: list[str]) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for p in parquet_paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def _filter_gradable(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep only instances the installed harness can build a test spec for."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.test_spec import make_test_spec

    good, dropped = [], []
    for e in rows:
        r, v = e.get("repo"), e.get("version")
        if r in MAP_REPO_VERSION_TO_SPECS and v in MAP_REPO_VERSION_TO_SPECS[r]:
            try:
                make_test_spec(e)
                good.append(e)
                continue
            except Exception as exc:  # spec present but build fails
                dropped.append(f"{e.get('instance_id')}: {exc}")
        else:
            dropped.append(f"{e.get('instance_id')}: (repo,version)=({r},{v!r}) not in spec map")
    return good, dropped


def _stratified_split(rows: list[dict], val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Split off a validation set whose repo mix MIRRORS the training set.

    The old split was ``good[:val_size]``. SWE-bench Verified's parquet is
    ordered by repo, so that took the first 20 astropy instances and nothing
    else: val was 100% astropy while train was 48% django and held 2 astropy
    instances. Validation reward and validation latency were therefore measuring
    a different workload than the one being trained on, and neither number was
    comparable to the training batch.

    Proportional (largest-remainder) allocation over repos, so a repo with 46% of
    the data gets ~46% of the val slots and repos too small to earn one get none
    -- which is the right behaviour here: val should look like train, not like a
    uniform tour of the repos. Seeded, so the split is reproducible.
    """
    by_repo: dict[str, list[dict]] = {}
    for r in rows:
        by_repo.setdefault(r.get("repo") or "?", []).append(r)

    # Hamilton apportionment: floor the exact quota, then hand out the leftover
    # seats to the largest remainders.
    total = len(rows)
    exact = {k: len(v) * val_size / total for k, v in by_repo.items()}
    quota = {k: int(v) for k, v in exact.items()}
    left = val_size - sum(quota.values())
    for k in sorted(by_repo, key=lambda k: (-(exact[k] - quota[k]), k))[:left]:
        quota[k] += 1

    rng = random.Random(seed)
    val, train = [], []
    for repo in sorted(by_repo):
        pool = sorted(by_repo[repo], key=lambda r: r.get("instance_id") or "")
        picked = set(rng.sample(range(len(pool)), quota[repo])) if quota[repo] else set()
        for i, row in enumerate(pool):
            (val if i in picked else train).append(row)
    rng.shuffle(train)
    return val, train


def _to_row(example: dict) -> dict:
    return {
        "data_source": "swebench",
        "agent_name": "swebench_agent",
        "prompt": [{"role": "user", "content": example.get("problem_statement", "")}],
        "reward_model": {"style": "rule", "ground_truth": example["instance_id"]},
        "extra_info": dict(example),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="data/swebench_verified")
    ap.add_argument("--val-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the stratified train/val split (see _stratified_split).")
    ap.add_argument("--proxy", default=DEFAULT_PROXY, help=f"Egress HTTP proxy (default {DEFAULT_PROXY}).")
    ap.add_argument("--keep-ungradable", action="store_true",
                    help="Do not drop instances the harness can't grade (NOT recommended).")
    args = ap.parse_args()

    proxy = _set_proxy(args.proxy)
    print(f"Dataset : {args.dataset} (split={args.split})")
    print(f"Proxy   : {proxy or '(none)'}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        parquets = _download_split_parquets(args.dataset, args.split, proxy, tmp)
        rows = _load_rows(parquets)
    print(f"Downloaded {len(rows)} raw instances.")

    if args.keep_ungradable:
        good = rows
    else:
        good, dropped = _filter_gradable(rows)
        if dropped:
            print(f"Dropped {len(dropped)} ungradable instances (repo/version not in installed swebench spec map).")
            for d in dropped[:5]:
                print("   -", d)
        print(f"Gradable: {len(good)}/{len(rows)}")
    if not good:
        raise SystemExit("No gradable instances — check the swebench version vs. the dataset.")

    os.makedirs(args.out_dir, exist_ok=True)
    val_size = min(args.val_size, len(good))
    val, train = _stratified_split(good, val_size, args.seed)

    import pyarrow as pa
    import pyarrow.parquet as pq

    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val.parquet")
    pq.write_table(pa.Table.from_pylist([_to_row(e) for e in train]), train_path)
    pq.write_table(pa.Table.from_pylist([_to_row(e) for e in val]), val_path)
    print(f"\nWrote:\n  {train_path}  ({len(train)} train)\n  {val_path}  ({len(val)} val)")

    from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name  # type: ignore

    print("\nSample instance images (for the docker pull/scp step):")
    for e in good[:3]:
        print("  ", e["instance_id"], "->", get_swebench_docker_image_name(e))


if __name__ == "__main__":
    main()
