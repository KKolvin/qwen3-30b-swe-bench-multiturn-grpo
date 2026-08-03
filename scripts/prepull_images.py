#!/usr/bin/env python3
"""Pre-pull SWE-bench images from the 1ms.run mirror and cache them locally under
the `swebench/<img>:latest` name the rollout containers expect.

Why: daocloud (mirror #1) 403s the `swebench/*` namespace and direct Docker Hub
times out; only docker.1ms.run serves these images. It rate-limits per IP (429),
so we pull gently (small concurrency + exponential backoff) rather than stampede.

IMPORTANT — must target the ROOTLESS daemon: training runs containers on the
rootless docker (see agent_loop.py), whose image store lives on /data1
(/data1/<user>/docker). The rootful daemon's store is on `/`, which is tiny and
fills instantly. We default DOCKER_HOST to the rootless socket if it isn't set,
and the disk guard watches BOTH the store filesystem and `/` so a misconfigured
daemon can't silently fill root.

Idempotent: skips images already tagged locally. Safe to re-run for a second pass
over failures. Reads the missing-image list from logs/missing_images.txt.
"""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Target the rootless daemon by default (its store is on /data1, not `/`).
_rootless_sock = f"/run/user/{os.getuid()}/docker.sock"
if "DOCKER_HOST" not in os.environ and os.path.exists(_rootless_sock):
    os.environ["DOCKER_HOST"] = f"unix://{_rootless_sock}"

MIRROR = "docker.1ms.run"
# Full set of images the target daemon should hold (local_has() below skips any
# already cached, so feeding the complete list only pulls what's absent).
LIST = Path("logs/verified_all_images.txt")
FAIL_LOG = Path("logs/prepull_failures.txt")
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAX_ATTEMPTS = 8
PULL_TIMEOUT = 900        # kill+retry a single `docker pull` that stalls this long (s)
# Per-mount free floors (GB). /data1 is the rootless image store — needs real
# headroom. `/` is only a catastrophe floor: rootless pulls never write there, so
# it should sit flat at whatever it already is; a low floor still catches a
# wrong-daemon pull dumping images onto root. `/` legitimately runs ~55GB free,
# so its floor MUST stay well below that or every pull is wrongly skipped.
GUARD_MOUNTS = {"/data1": 80, "/": 10}


def free_gb(path: str) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / 1e9


def low_mount() -> str | None:
    """Return the first guarded mount below its free floor, else None."""
    for m, floor in GUARD_MOUNTS.items():
        try:
            if free_gb(m) < floor:
                return m
        except OSError:
            continue  # mount absent on this host; skip
    return None


def local_has(img: str) -> bool:
    r = subprocess.run(["docker", "image", "inspect", f"swebench/{img}:latest"],
                       capture_output=True)
    return r.returncode == 0


def pull_one(img: str) -> tuple[str, str]:
    """Returns (img, status) where status in {cached, ok, disk, failed}."""
    if local_has(img):
        return img, "cached"
    src = f"{MIRROR}/swebench/{img}:latest"
    dst = f"swebench/{img}:latest"
    backoff = 60
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if low_mount() is not None:
            return img, "disk"
        # `docker pull` has no built-in timeout: a stalled download hangs forever
        # and permanently occupies a worker slot. Bound each attempt and treat a
        # timeout as transient so it retries (with a fresh pull) after backoff.
        try:
            p = subprocess.run(["docker", "pull", src], capture_output=True,
                               text=True, timeout=PULL_TIMEOUT)
            out = (p.stdout + p.stderr)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rmi", "-f", src], capture_output=True)  # drop partial
            out, rc = f"pull timeout after {PULL_TIMEOUT}s", 1
        if rc == 0:
            subprocess.run(["docker", "tag", src, dst], check=False)
            subprocess.run(["docker", "rmi", src], capture_output=True)  # drop mirror tag, keep layers
            return img, "ok"
        transient = ("429" in out or "timeout" in out.lower() or "i/o" in out.lower()
                     or "eof" in out.lower() or "unknown blob" in out.lower()
                     or "connection" in out.lower())
        if attempt < MAX_ATTEMPTS and transient:
            time.sleep(min(backoff, 300))
            backoff *= 2
            continue
        # Mirror doesn't have this tag ("manifest unknown"/"not found") but Docker
        # Hub might. Fall back to a bare `docker pull` (daemon mirror-fallback ->
        # docker.io) once before giving up. This lands directly on `dst`.
        if "manifest unknown" in out.lower() or "not found" in out.lower():
            try:
                p = subprocess.run(["docker", "pull", dst], capture_output=True,
                                   text=True, timeout=PULL_TIMEOUT)
                if p.returncode == 0:
                    return img, "ok"
                out = p.stdout + p.stderr
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rmi", "-f", dst], capture_output=True)
                out = f"direct-pull timeout after {PULL_TIMEOUT}s"
        FAIL_LOG.open("a").write(f"{img}\t{out.strip()[-160:]}\n")
        return img, "failed"
    return img, "failed"


# Images that cannot be pulled/run under the current rootless config: their
# layers contain files with UIDs beyond the 65536 subuid range, so extraction
# fails with `lchown ... invalid argument`. Skip them (they only waste ~1.5GB of
# download each before failing). Remove from this set once the admin enlarges the
# subuid/subgid range. Override with PREPULL_SKIP="" to force-attempt them.
SKIP_SUBSTR = tuple(s for s in os.environ.get("PREPULL_SKIP", "matplotlib").split(",") if s)


def main():
    imgs = [l.strip() for l in LIST.read_text().splitlines() if l.strip()]
    skipped = [i for i in imgs if any(s in i for s in SKIP_SUBSTR)]
    imgs = [i for i in imgs if i not in skipped]
    todo = [i for i in imgs if not local_has(i)]
    done = len(imgs) - len(todo)
    print(f"[prepull] {len(imgs)} pullable ({len(skipped)} skipped: {','.join(SKIP_SUBSTR)}); "
          f"{done} already cached; pulling {len(todo)} via {MIRROR} "
          f"(concurrency={CONCURRENCY})", flush=True)
    counts = {"ok": 0, "cached": 0, "failed": 0, "disk": 0}
    n = 0
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(pull_one, i): i for i in todo}
        for fut in cf.as_completed(futs):
            img, status = fut.result()
            counts[status] += 1
            n += 1
            if status == "disk":
                print(f"[prepull] STOP: {low_mount()} fell below its free floor; "
                      f"{img} skipped. Cached so far: {counts['ok']}", flush=True)
            if n % 10 == 0 or status in ("failed", "disk"):
                free = "  ".join(f"{m}={free_gb(m):.0f}GB"
                                 for m in GUARD_MOUNTS if os.path.exists(m))
                print(f"[prepull] {n}/{len(todo)}  ok={counts['ok']} "
                      f"failed={counts['failed']} disk={counts['disk']}  {free}", flush=True)
    print(f"[prepull] DONE ok={counts['ok']} cached={counts['cached']} "
          f"failed={counts['failed']} disk={counts['disk']}", flush=True)
    if counts["failed"]:
        print(f"[prepull] failures logged to {FAIL_LOG} — re-run for another pass", flush=True)


if __name__ == "__main__":
    main()
