"""What only developing the simulator needs: our own runs, and their answers.

At use time (`SIMULATOR.md` §1) the input is somebody else's trace and a config
bundle, with no timeline. Everything here assumes one of our runs instead: the
adapter for our own timeline shards, which carry the answers next to the
workload, and the code that measures priors off a run to build
``default_priors.json`` or to run the degradation test of §15.

Nothing outside this package may import it, and a test holds that line
(``tests/test_simulator_ir.py``): a use-time path that reaches in here only works
on our runs, and would never be found out while developing on them. The
dependency points one way: this package uses :mod:`simulator.adapters`, never
the other way round.
"""
