#!/usr/bin/env python
"""Print the ``priors`` section of a config bundle: every knob, what it measures,
our run's median for it, and how much of the task time it accounts for.

    python scripts/priors_template.py >> my_bundle.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulator.adapters.priors import priors_template  # noqa: E402

print(priors_template(), end="")
