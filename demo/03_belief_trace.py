#!/usr/bin/env python3
"""DEMO 3 - Watch the baseline's belief move as evidence arrives.
(~5 seconds, no GPU, no weights)

Prints, round by round, what the Bayesian baseline believes and what it
therefore decides. This is the target that Probe 1 is trained to decode from
the model's residual stream, so it is worth seeing directly.

    python demo/03_belief_trace.py                 # stag_hunt, seed 0
    python demo/03_belief_trace.py trust_game 3

Columns:
    t          decision step
    observed   what the scripted behaviour policy did and what came back
    posterior  p(theta | h_t) over the hidden types
    p(act1)    the baseline's score for action 1
    optimal    the resulting optimal action; "~" marks a near-indifferent step
               (|p-0.5| <= 0.10), which is excluded from every reported metric
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import ENVS, make_env                     # noqa: E402

T = 10
AMBIG = 0.10


def type_names(env, name):
    mod = importlib.import_module("envs." + name)
    return (getattr(type(env), "TYPES", None)
            or getattr(mod, "TYPES", None)
            or getattr(mod, "RULES", None))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "stag_hunt"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if name not in ENVS:
        sys.exit(f"unknown scenario {name!r}; choose from {list(ENVS)}")

    env = make_env(name, np.random.default_rng(seed), T, seed=seed)
    types = type_names(env, name)
    has_post = hasattr(env, "posterior")

    print("=" * 78)
    print(f"BELIEF TRAJECTORY  --  {name}, seed {seed}")
    print("=" * 78)
    print(f"hidden parameter (never shown to the model): {env.hidden_param()}")
    if types:
        print(f"candidate hidden types (uniform prior)     : {', '.join(types)}")
    print(f"action 0 = {env.act0}   action 1 = {env.act1}")
    print()

    head = f"{'t':>2}  {'observed':<34}"
    if has_post and types:
        head += "  " + "".join(f"{ty[:9]:>10}" for ty in types)
    head += f"{'p(act1)':>10}  optimal"
    print(head)
    print("-" * len(head))

    records = []
    for t in range(1, T + 1):
        # belief BEFORE this round's outcome is revealed
        p = env.oracle_p_act1(records)
        row = f"{t:>2}  "
        row += f"{'(prior, no evidence yet)' if not records else obs:<34}"
        if has_post and types:
            post = np.asarray(env.posterior(records), dtype=float)
            row += "  " + "".join(f"{v:>10.3f}" for v in post)
        opt = env.act1 if p > 0.5 else env.act0
        mark = "~" if abs(p - 0.5) <= AMBIG else " "
        row += f"{p:>10.3f}  {opt}{mark}"
        print(row)

        ctx = env.new_round(t)
        a = env.behavior_action(t)
        _, rec = env.outcome(t, ctx, a)
        records.append(rec)
        obs = ", ".join(f"{k}={v}" for k, v in rec.items() if k != "t")[:34]

    print()
    print("~ marks a near-indifferent step, dropped from all reported metrics.")
    print("Probe 1 is trained to decode the sign of the utility margin behind")
    print("the p(act1) column from the model's residual stream at this step.")


if __name__ == "__main__":
    main()
