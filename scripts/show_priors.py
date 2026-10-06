#!/usr/bin/env python
"""Show what each prior knob uses under a config bundle: the user's number or our run's.

    python scripts/show_priors.py configs/simulator/example_bundle.yaml
    python scripts/show_priors.py            # no bundle: every default
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulator.adapters.base import AdapterError  # noqa: E402
from simulator.adapters.priors import describe_priors, load_priors  # noqa: E402

try:
    print(describe_priors(load_priors(Path(sys.argv[1]) if len(sys.argv) > 1 else None)), end="")
except AdapterError as e:
    sys.exit(f"error: {e}")
