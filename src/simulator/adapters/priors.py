"""Priors at use time: our run's frozen defaults, and what a config bundle states over them.

A trace from an unfamiliar harness leaves some work untimed, and every field it
leaves out needs a distribution somebody measured. By default that somebody is
us: ``default_priors.json`` next to this file is a frozen snapshot of one step of
one of our runs, with a scale stretched across our other runs. Measuring and
writing it is development work, done by :mod:`simulator.dev.priors` and
``scripts/simulator_dev/build_default_priors.py``; this module only reads it. A
config bundle that states a knob replaces that knob's default, and
:func:`resolve_priors` is where the two meet (`SIMULATOR.md` §5).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from simulator.adapters.base import KNOB_BY_KIND, AdapterError, Prior, Priors
from simulator.ir import Dist, Empirical, Uniform

#: The snapshot :func:`default_priors` reads.
DEFAULT_PRIORS_PATH = Path(__file__).with_name("default_priors.json")

#: What each knob measures, in the words the config template shows a user.
KNOB_HELP: Mapping[str, str] = {
    "eval_duration": "running the tests that score one submitted patch",
    "tool_duration": "one tool call (bash, edit, submit, ...)",
    "container_start": "starting the container one episode runs in",
    "patch_recover": "pulling the diff out of an episode that never submitted",
    "cleanup": "removing the container when an episode ends",
}


# ---------------------------------------------------------------------------
# the default: our run, frozen to a file
# ---------------------------------------------------------------------------
def priors_from_json(doc: Mapping[str, Any], *, default: bool = False) -> Priors:
    """Inverse of :func:`simulator.dev.priors.priors_to_json`. ``default`` marks every prior as the simulator's default."""
    out = []
    for knob, d in doc["priors"].items():
        scale = Uniform(*d["scale"]) if d.get("scale") else None
        out.append(Prior(knob=knob, dist=Empirical(tuple(d["values"])), source=d["source"],
                         scale=scale, scale_sources=tuple(d.get("scale_sources", ())), default=default))
    return Priors.of(*out)


@lru_cache(maxsize=1)
def default_priors() -> Priors:
    """Our run's distributions, each marked :attr:`~Prior.default`. See ``default_priors.json``'s ``meta``."""
    return priors_from_json(json.loads(DEFAULT_PRIORS_PATH.read_text()), default=True)


def default_priors_meta() -> dict[str, Any]:
    """Which runs the defaults came from and how they were built, for the provenance report."""
    return json.loads(DEFAULT_PRIORS_PATH.read_text())["meta"]


# ---------------------------------------------------------------------------
# what the config bundle states, over the default
# ---------------------------------------------------------------------------
def resolve_priors(stated: Mapping[str, Any] | None, *, where: str,
                   defaults: Priors | None = None) -> Priors:
    """The priors a conversion uses: what the bundle states, and our run for everything else.

    ``stated`` is the bundle's ``priors`` section, and a user states one thing per
    knob, the median seconds per occurrence, because that is the number people
    know about their own harness::

        eval_duration: 40          # a patch takes about 40 s to score
        tool_duration: [0.5, 2]    # a tool call takes 0.5 to 2 s, not sure where

    The number moves our run's distribution so its median lands there; how much
    one occurrence differs from the next stays what we measured, since nobody
    knows that about a harness they did not trace. A single number is a claim
    that the median is right, so that knob gets no scale. A range is the scale:
    the median is drawn between ``lo`` and ``hi``, once per realisation, which is
    what the sensitivity table sweeps (`SIMULATOR.md` §6). Our run's own scale is
    dropped either way: it measures how far our runs drift, not how far off
    someone else's number is. ``where`` names the bundle, for the error messages.

    Unstated knobs keep the default whole, marked :attr:`~Prior.default`. A
    knob nobody reads is refused, since a typo would otherwise leave the default
    in place while the user believes they replaced it. :func:`priors_template`
    lists the knobs.
    """
    defaults = default_priors() if defaults is None else defaults
    stated = dict(stated or {})
    known = set(KNOB_BY_KIND.values()) | set(defaults.by_knob)
    stray = sorted(set(stated) - known)
    if stray:
        raise AdapterError(f"{where}: priors {stray} name no knob; known knobs: {sorted(known)}")
    out = []
    for knob in sorted(known):
        if knob in stated:
            out.append(_stated(knob, stated[knob], defaults.by_knob.get(knob), f"{where}: prior {knob!r}"))
        elif knob in defaults.by_knob:
            out.append(defaults.by_knob[knob])
    return Priors.of(*out)


def load_priors(bundle: Path | None) -> Priors:
    """The priors of a config bundle file: its ``priors`` section over our defaults.

    ``None``, an empty file, or a bundle with no ``priors`` section all mean every
    default. Other sections of the bundle are not read here.
    """
    if bundle is None:
        return resolve_priors(None, where="no bundle")
    doc = yaml.safe_load(Path(bundle).read_text()) or {}
    if not isinstance(doc, Mapping):
        raise AdapterError(f"{bundle}: a config bundle is a mapping of sections; got {type(doc).__name__}")
    stated = doc.get("priors") or {}
    if not isinstance(stated, Mapping):
        raise AdapterError(f"{bundle}: 'priors' must map knob names to seconds; got {stated!r}")
    return resolve_priors(stated, where=str(bundle))


def describe_priors(priors: Priors) -> str:
    """One line per knob: the median and range it will draw, and whose number that is."""
    lines = [f"{'knob':16} {'median':>9}  {'range of the median':20} from"]
    for knob, p in sorted(priors.by_knob.items()):
        m = p.dist.quantile(0.5)
        at = (lambda q: m * p.scale.quantile(q)) if p.scale else (lambda q: m)  # noqa: E731
        rng = f"{at(0.0):.3g} - {at(1.0):.3g}" if p.scale else "fixed"
        m = at(0.5)
        who = f"our run ({p.source})" if p.default else f"you (shape from {p.source})" \
            if p.source.startswith("run:") else f"you ({p.source})"
        lines.append(f"{knob:16} {m:9.3g}  {rng:20} {who}")
    return "\n".join(lines) + "\n"


def _stated(knob: str, spec: Any, default: Prior | None, what: str) -> Prior:
    if _is_number(spec):
        lo = hi = _seconds(spec, what)
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = (_seconds(x, what) for x in spec)
        if lo > hi:
            raise AdapterError(f"{what} has lo {lo} above hi {hi}")
    else:
        raise AdapterError(f"{what} must be the median seconds, as a number or [lo, hi]; got {spec!r}")
    scale = Uniform(lo / hi, 1.0) if lo < hi else None
    if default is None:
        # No shape of ours to move: a number is every occurrence, a range spreads them.
        return Prior(knob=knob, dist=Empirical((hi,)) if lo == hi else Uniform(lo, hi), source=f"bundle:{what}")
    m = default.dist.quantile(0.5)
    if m <= 0:
        raise AdapterError(f"{what}: our run's median for this knob is {m}, so it cannot be moved to {hi}")
    if isinstance(default.dist, Empirical):
        dist: Dist = Empirical(tuple(v * hi / m for v in default.dist.values))
    else:
        dist = Uniform(default.dist.lo * hi / m, default.dist.hi * hi / m)  # type: ignore[attr-defined]
    # The source stays our run: the shape is still borrowed from it, so check
    # still refuses to convert that run with it.
    return replace(default, dist=dist, scale=scale, scale_sources=(), default=False)


def priors_template(defaults: Priors | None = None) -> str:
    """The ``priors`` section of a config bundle, every knob commented out at our run's value.

    Ordered by each knob's share of task time in our run, so the top lines are
    the ones worth replacing: the last ones barely move an end-to-end time.
    """
    defaults = default_priors() if defaults is None else defaults
    total = {k: len(p.dist.values) * _mean(p.dist) for k, p in defaults.by_knob.items()
             if isinstance(p.dist, Empirical)}
    whole = sum(total.values()) or 1.0
    runs = sorted({p.source.split("/", 1)[0] for p in defaults.by_knob.values()})
    lines = [
        "# How long the work the trace did not time takes, in seconds.",
        "# Uncomment a knob to replace our number; anything left commented uses our run",
        f"# ({', '.join(runs)}).",
        "#   knob: 2          median seconds per occurrence",
        "#   knob: [1, 4]     not sure: the median is somewhere in here",
        "# Your number moves our run's distribution; the spread between occurrences stays ours.",
        "# Ordered by share of task time in our run: the top lines matter, the bottom ones barely do.",
        "priors:",
    ]
    for knob in sorted(defaults.by_knob, key=lambda k: -total.get(k, 0.0)):
        p = defaults.by_knob[knob]
        value = f"{p.dist.quantile(0.5):.3g}"
        share = f"{100 * total[knob] / whole:.1f}% of task time" if knob in total else ""
        drift = f", our runs drifted {p.scale.lo:.2f}-{p.scale.hi:.2f}x" if isinstance(p.scale, Uniform) else ""
        lines.append(f"  # {f'{knob}: {value}':<26} # {KNOB_HELP.get(knob, '')}. {share}{drift}")
    return "\n".join(lines) + "\n"


def _seconds(x: Any, what: str) -> float:
    if not _is_number(x) or not math.isfinite(x) or x < 0:
        raise AdapterError(f"{what}: {x!r} is not a finite non-negative number of seconds")
    return float(x)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(d: Dist) -> float:
    xs = d.values  # type: ignore[attr-defined]
    return sum(xs) / len(xs) if xs else float("nan")
