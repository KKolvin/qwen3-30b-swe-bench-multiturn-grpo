#!/usr/bin/env python
"""Prepare a ONE-instance dataset for a single-image, single-step smoke test.

Given an instance_id (or a row index into the val set), write
``data/swebench/one.parquet`` containing just that instance, and print the exact
docker image name mini-swe-agent will request for it plus the pull/save/scp/load
recipe for getting that image onto an air-gapped training box.

Usage:
    python scripts/prepare_one.py --instance-id pandas-dev__pandas-10007
    python scripts/prepare_one.py --index 0            # first val instance
    python scripts/prepare_one.py --index 0 --image-only   # just print the image name
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_grpo.config import DataConfig  # noqa: E402


def _image_name(instance: dict) -> str:
    from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name  # type: ignore

    return get_swebench_docker_image_name(instance)


def _find_instance(instance_id: str | None, index: int | None) -> dict:
    import pyarrow.parquet as pq

    cfg = DataConfig()
    rows = []
    for name in ("val.parquet", "train.parquet"):
        path = os.path.join(cfg.output_dir, name)
        if os.path.isfile(path):
            rows.extend(pq.read_table(path).to_pylist())
    if not rows:
        raise SystemExit(f"No parquet under {cfg.output_dir}. Run scripts/prepare_swebench_hf.py first.")

    if instance_id is not None:
        for r in rows:
            if r["extra_info"]["instance_id"] == instance_id:
                return r
        raise SystemExit(f"instance_id {instance_id!r} not found in val/train parquet.")

    idx = index or 0
    if idx >= len(rows):
        raise SystemExit(f"--index {idx} out of range ({len(rows)} rows).")
    return rows[idx]


def _print_recipe(image: str) -> None:
    tar = image.split("/")[-1].replace(":", "_") + ".tar.gz"
    print("\n" + "=" * 72)
    print("DOCKER IMAGE FOR THIS INSTANCE:")
    print(f"    {image}")
    print("=" * 72)
    print("On a host WITH docker.io access, export the image:")
    print(f"    docker pull {image}")
    print(f"    docker save {image} | gzip > {tar}")
    print("Copy it to this box and load it:")
    print(f"    scp {tar} <this-host>:{os.getcwd()}/")
    print(f"    gunzip -c {tar} | docker load")
    print("Verify it landed:")
    print(f"    docker image inspect {image} >/dev/null && echo OK")
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--instance-id", default=None)
    g.add_argument("--index", type=int, default=None)
    ap.add_argument("--image-only", action="store_true", help="Only print the image name + recipe; don't write parquet.")
    ap.add_argument("--out", default=None, help="Output parquet path (default data/swebench/one.parquet).")
    args = ap.parse_args()

    row = _find_instance(args.instance_id, args.index)
    inst = row["extra_info"]
    image = _image_name(inst)
    print(f"instance_id: {inst['instance_id']}  repo: {inst['repo']}")
    _print_recipe(image)

    if args.image_only:
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = args.out or os.path.join(DataConfig().output_dir, "one.parquet")
    pq.write_table(pa.Table.from_pylist([row]), out)
    print(f"Wrote 1-instance dataset -> {out}")
    print("Then run one step:")
    print(f"    python scripts/train_standalone.py --train {out} --batch-size 1 --group-size 2 --total-steps 1")


if __name__ == "__main__":
    main()
