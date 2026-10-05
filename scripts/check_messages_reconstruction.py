#!/usr/bin/env python
"""Can the conversation text be turned back into the token counts the server saw?

Reads a run's ``messages-<pid>.jsonl`` (written with
``AGENTIC_TRAJECTORY_DUMP_MESSAGES=1``), re-renders every prompt with the chat
template and tool schemas from the file's header exactly as the agent loop did
(verl's ``apply_chat_template``, then the ``prompt_length`` left-truncation), and
compares the per-turn ``prompt_tokens`` / ``completion_tokens`` with what the
dump recorded for the same episode. This is the §14 assumption of
``SIMULATOR.md`` -- "tokenizer + template rebuilds the token counts" -- as a
number rather than a hope.

    python scripts/check_messages_reconstruction.py --dir /data0/shared/$USER/agentic-trajectories/<run>

Pairing is positional per worker file: the i-th episode line of
``messages-<pid>.jsonl`` is the i-th line of ``trajectories-<pid>.jsonl``, since
``Episode.flush`` writes both in one go. Instance id and exit status are checked.

Known, irreducible residual: the model sometimes samples a non-canonical token
split (" \\n" as two tokens where the tokenizer would merge them), so
re-encoding its text comes out a few tokens *longer*; that error carries into
every later prompt of the episode. First measurement (run 20261005-113503,
144 episodes): 93% of prompts and 98.7% of completions exact, the rest within
+1..+9 tokens.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="trajectory dump directory of one run")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many episodes (0 = all)")
    ap.add_argument("--show", type=int, default=5, help="print this many mismatching turns")
    ap.add_argument("--prompt-length", type=int, default=None,
                    help="rollout.prompt_length when the header predates the field (the run's hydra "
                         "config has it as data.max_prompt_length)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from verl.utils.tokenizer.chat_template import apply_chat_template, initialize_system_prompt

    pairs, meta = [], None
    for mp in sorted(glob.glob(os.path.join(args.dir, "messages-*.jsonl"))):
        pid = mp.rsplit("-", 1)[1]
        recs = [json.loads(l) for l in open(mp) if l.strip()]
        if not recs or recs[0].get("kind") != "meta":
            print(f"{mp}: no meta header, skipped", file=sys.stderr)
            continue
        meta = meta or recs[0]
        eps = [r for r in recs[1:] if r.get("kind") == "episode"]
        dump_path = os.path.join(args.dir, f"trajectories-{pid}")
        dumps = [json.loads(l) for l in open(dump_path) if l.strip()] if os.path.exists(dump_path) else []
        for e, d in zip(eps, dumps):
            if e["instance_id"] != d["instance_id"] or e["exit_status"] != d["exit_status"]:
                print(f"{mp}: positional pairing broke at {e['traj']}", file=sys.stderr)
                break
            pairs.append((e, d))
    if meta is None:
        print("no messages files found", file=sys.stderr)
        return 2
    if args.limit:
        pairs = pairs[: args.limit]

    tok = AutoTokenizer.from_pretrained(meta["model"])
    if tok.chat_template != meta["chat_template"]:
        print("WARNING: the tokenizer's chat template differs from the one in the file; using the file's")
        tok.chat_template = meta["chat_template"]
    tools = meta["tool_schemas"]
    cap = meta.get("prompt_length") or args.prompt_length
    if cap is None:
        print("header has no prompt_length and --prompt-length not given: first prompts over the cap "
              "will show as large positive diffs", file=sys.stderr)
    sys_n = len(initialize_system_prompt(tok))

    def render(msgs, with_tools):
        ids = apply_chat_template(tok, msgs, tools=tools if with_tools else None,
                                  add_generation_prompt=True, tokenize=True)
        return list(ids)

    stats: collections.Counter = collections.Counter()
    hist = {"prompt": collections.Counter(), "completion": collections.Counter()}
    shown = 0
    for e, d in pairs:
        msgs, turns = e["messages"], d["metrics"].get("turns") or []
        if len(msgs) < 2 or msgs[0]["role"] != "system":
            stats["odd_episodes"] += 1
            continue
        ids = render(msgs[:2], True)
        if cap and len(ids) > cap:
            ids = ids[-cap:]
            stats["first_prompt_capped"] += 1
        i, k = 2, 0
        drift = 0  # summed completion diff so far; the prompt should be off by exactly this
        repeats: collections.Counter = collections.Counter()
        while i < len(msgs) and k < len(turns):
            if msgs[i]["role"] != "assistant":
                stats["odd_episodes"] += 1
                break
            rec = turns[k]
            comp = tok.encode(msgs[i]["content"], add_special_tokens=False)
            dp, dc = len(ids) - rec["prompt_tokens"], len(comp) - rec["completion_tokens"]
            hist["prompt"][dp] += 1
            hist["completion"][dc] += 1
            stats["turns"] += 1
            if k == 0:
                first_dp = dp
            elif dp != first_dp + drift:
                stats["observation_mismatch"] += 1
                if shown < args.show:
                    shown += 1
                    print(f"  {e['traj']} turn {k + 1}: prompt off by {dp:+d} but completions so far "
                          f"account for {first_dp + drift:+d}; an observation re-rendered differently")
            drift += dc
            if dc:
                repeats[msgs[i]["content"][-80:]] += 1
            if dp or dc:
                if shown < args.show:
                    shown += 1
                    print(f"  {e['traj']} turn {k + 1}: prompt {len(ids)} vs {rec['prompt_tokens']} ({dp:+d}), "
                          f"completion {len(comp)} vs {rec['completion_tokens']} ({dc:+d})")
            ids += comp
            i += 1
            k += 1
            obs = []
            while i < len(msgs) and msgs[i]["role"] != "assistant":
                obs.append(msgs[i])
                i += 1
            if obs:
                obs_ids = render(obs, False)[sys_n:]
                if cap and len(obs_ids) > cap:
                    obs_ids = obs_ids[-cap:]
                ids += obs_ids
        stats["episodes"] += 1
        if k != len(turns):
            stats["turn_count_mismatch"] += 1
        for tail, n in repeats.items():
            if n >= 10:
                stats["repeated_drifting_turns"] += n
                if shown < args.show + 2:
                    shown += 1
                    print(f"  {e['traj']} ({e['exit_status']}): {n} turns end the same way and all "
                          f"re-encode off: {tail!r}")

    n = stats["turns"] or 1
    print(f"episodes {stats['episodes']}, turns {stats['turns']}, first prompts capped at {cap}: "
          f"{stats['first_prompt_capped']}, odd episodes {stats['odd_episodes']}, "
          f"turn-count mismatches {stats['turn_count_mismatch']}")
    print(f"observation renders that did not match: {stats['observation_mismatch']} "
          f"(0 means every prompt error is the first prompt's cap plus completion re-encoding drift); "
          f"drifting turns inside repeating episodes: {stats['repeated_drifting_turns']}")
    for name, c in hist.items():
        exact = c[0]
        print(f"{name:10s} exact {exact}/{n} ({100 * exact / n:.1f}%); diff histogram (rebuilt - recorded): "
              f"{sorted(c.items())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
