#!/usr/bin/env python3
"""Repack a SWE-bench image so it can be loaded under rootless docker WITHOUT a
larger subuid range.

Why: matplotlib images contain files owned by Windows-origin UID/GID 197609/197121,
far beyond the user's 65536 subuid range. Rootless fuse-overlayfs can't map that ->
`lchown ... invalid argument` at extraction, so `docker pull` fails. This tool pulls
the layer blobs straight from the registry (no daemon extraction), rewrites every
tar entry with uid/gid > 65535 down to 0:0 (root-in-container; harmless for running
tests), recomputes each layer's diffID, patches the image config to match, assembles
a `docker save`-format archive, and `docker load`s it.

Zero system root. Keeps the fuse-overlayfs store and the existing image cache intact.

Usage: python3 scripts/repack_image.py <repo>   e.g.
       python3 scripts/repack_image.py swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-13989
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REGISTRY = "docker.1ms.run"
TOKEN_URL = f"https://{REGISTRY}/openapi/v1/auth/token"
ACCEPT = ("application/vnd.docker.distribution.manifest.v2+json,"
          "application/vnd.oci.image.manifest.v1+json")
MAX_ID = 65535           # anything above this has no rootless mapping -> squash to 0
WORK_ROOT = Path("/data1/shared/kewen_repack")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def token(repo: str) -> str:
    r = sh(["curl", "-sS", f"{TOKEN_URL}?service={REGISTRY}&scope=repository:{repo}:pull"])
    return json.loads(r.stdout).get("token") or json.loads(r.stdout)["access_token"]


def fetch_json(repo: str, ref: str, kind: str) -> dict:
    url = f"https://{REGISTRY}/v2/{repo}/{kind}/{ref}"
    r = sh(["curl", "-sS", "-H", f"Authorization: Bearer {token(repo)}",
            "-H", f"Accept: {ACCEPT}", url])
    return json.loads(r.stdout)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def fetch_blob(repo: str, digest: str, dest: Path) -> None:
    """Download a blob to dest and VERIFY its sha256 matches `digest`. Fresh token
    per attempt (mirror tokens are short-lived and big blobs outlast them). The
    mirror both drops HTTP/2 streams (curl 92) AND occasionally returns a small JSON
    error body with a success code — a size>0 check accepts that garbage and gzip
    parsing then fails. Digest verification is the only reliable gate: it rejects
    error bodies, truncation, and corruption alike, and we retry until bytes match."""
    if dest.exists() and sha256_file(dest) == digest:
        print(f"    have {digest[:19]} ({dest.stat().st_size/1e6:.0f}MB)", flush=True)
        return
    url = f"https://{REGISTRY}/v2/{repo}/blobs/{digest}"
    backoff = 5
    for attempt in range(1, 7):
        r = subprocess.run(
            ["curl", "-sSL", "--http1.1", "--retry", "3", "--retry-all-errors",
             "--connect-timeout", "30", "-H", f"Authorization: Bearer {token(repo)}",
             "-o", str(dest), url])
        if r.returncode == 0 and dest.exists() and sha256_file(dest) == digest:
            print(f"    got  {digest[:19]} ({dest.stat().st_size/1e6:.0f}MB)", flush=True)
            return
        bad = "digest mismatch" if (dest.exists() and dest.stat().st_size) else f"curl rc={r.returncode}"
        dest.unlink(missing_ok=True)
        print(f"    retry {attempt}/6 {digest[:19]} ({bad})", flush=True)
        time.sleep(min(backoff, 60))
        backoff *= 2
    raise RuntimeError(f"blob download failed after retries: {digest}")


def rewrite_layer(gz_path: Path, out_tar: Path) -> tuple[str, int]:
    """Read a gzipped layer, write an UNCOMPRESSED tar with all uid/gid>MAX_ID
    clamped to 0. Return (diffID = sha256 of the uncompressed tar, num_fixed)."""
    fixed = 0
    with tarfile.open(out_tar, mode="w") as tout, \
         tarfile.open(gz_path, mode="r:gz") as tin:
        for m in tin:
            if m.uid > MAX_ID or m.gid > MAX_ID:
                fixed += 1
                if m.uid > MAX_ID:
                    m.uid, m.uname = 0, ""
                if m.gid > MAX_ID:
                    m.gid, m.gname = 0, ""
            data = tin.extractfile(m) if m.isreg() else None
            tout.addfile(m, data)
    # diffID = sha256 of the uncompressed layer tar (stream it back off disk)
    h = hashlib.sha256()
    with open(out_tar, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest(), fixed


def main():
    repo = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "latest"
    work = WORK_ROOT / repo.replace("/", "_")
    blobs = work / "blobs"
    layers_dir = work / "arch"
    blobs.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    print(f"[repack] {repo}:{tag}", flush=True)
    man = fetch_json(repo, tag, "manifests")
    cfg_digest = man["config"]["digest"]
    layer_digests = [l["digest"] for l in man["layers"]]
    print(f"[repack] config={cfg_digest[:19]}  {len(layer_digests)} layers", flush=True)

    # config blob
    cfg_blob = blobs / (cfg_digest.replace(":", "_") + ".json")
    fetch_blob(repo, cfg_digest, cfg_blob)
    config = json.loads(cfg_blob.read_text())

    # layers: download, rewrite, collect new diffIDs + archive filenames
    new_diff_ids, arch_layers, total_fixed = [], [], 0
    for i, dg in enumerate(layer_digests):
        print(f"[repack] layer {i+1}/{len(layer_digests)}", flush=True)
        gz = blobs / (dg.replace(":", "_") + ".tgz")
        fetch_blob(repo, dg, gz)
        out = layers_dir / f"layer_{i:02d}.tar"
        diff_id, fixed = rewrite_layer(gz, out)
        total_fixed += fixed
        if fixed:
            print(f"    clamped {fixed} out-of-range entries -> 0:0", flush=True)
        new_diff_ids.append(diff_id)
        arch_layers.append(out.name)

    # patch config: diff_ids must match the rewritten layers
    config["rootfs"]["diff_ids"] = new_diff_ids
    cfg_bytes = json.dumps(config).encode()
    cfg_name = hashlib.sha256(cfg_bytes).hexdigest() + ".json"
    (layers_dir / cfg_name).write_bytes(cfg_bytes)

    manifest = [{
        "Config": cfg_name,
        "RepoTags": [f"{repo}:{tag}"],
        "Layers": arch_layers,
    }]
    (layers_dir / "manifest.json").write_text(json.dumps(manifest))

    print(f"[repack] total entries clamped: {total_fixed}", flush=True)
    print(f"[repack] loading archive via docker load ...", flush=True)
    env = dict(os.environ, DOCKER_HOST=f"unix:///run/user/{os.getuid()}/docker.sock")
    tar = subprocess.Popen(["tar", "-c", "-C", str(layers_dir), "."], stdout=subprocess.PIPE)
    load = subprocess.run(["docker", "load"], stdin=tar.stdout, env=env,
                          capture_output=True, text=True)
    tar.wait()
    print(load.stdout + load.stderr, flush=True)
    if load.returncode != 0:
        print("[repack] FAILED (scratch kept for debugging)", flush=True)
        sys.exit(1)
    # free the ~5GB scratch (blobs + uncompressed layer tars) on success
    subprocess.run(["rm", "-rf", str(work)])
    print("[repack] OK", flush=True)


if __name__ == "__main__":
    main()
