#!/usr/bin/env python3
"""DEMO 1 - See exactly what the model is shown.  (~5 seconds, no GPU, no weights)

Builds a real scenario, runs the scripted behaviour policy for a few rounds, and
prints the resulting prompt verbatim: the same string, character for character,
that a model receives during the experiments.

    python demo/01_render_prompt.py                  # bandit, t=5
    python demo/01_render_prompt.py stag_hunt 8      # any scenario, any round

Read this first. Everything else in the repository is downstream of what this
prints.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import ENVS, make_env                                    # noqa: E402
from prompting import (ANSWER_FORMAT_BOXED, load_template, render,  # noqa: E402
                       rollout)

T = 10


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main():
    env_name = sys.argv[1] if len(sys.argv) > 1 else "bandit"
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if env_name not in ENVS:
        sys.exit(f"unknown scenario {env_name!r}; choose from {list(ENVS)}")

    rng = np.random.default_rng(0)
    env = make_env(env_name, rng, T, seed=0)
    records, ctxs = rollout(env, T, upto=t - 1)

    banner(f"SCENARIO {env_name}   round t={t} of {T}   seed 0")
    print(f"hidden parameter (never shown to the model): {env.hidden_param()}")
    print(f"action 0 = {env.act0:<10} {env.act0_desc}")
    print(f"action 1 = {env.act1:<10} {env.act1_desc}")

    banner("THE BASELINE'S ANSWER FOR THIS HISTORY")
    p = env.oracle_p_act1(records)
    print(f"P(action 1 is optimal | history) = {p:.4f}")
    print(f"optimal action                   = "
          f"{env.act1 if p > 0.5 else env.act0}")
    if abs(p - 0.5) <= 0.10:
        print("NOTE: |p-0.5| <= 0.10, so this step is treated as near-indifferent")
        print("      and is EXCLUDED from every reported metric.")

    ctx = env.new_round(t)
    history = env.format_history(records)

    banner("PROMPT SEEN BY PROBES 1 AND 2  (forced choice, mapping 0)")
    print(render(load_template(env_name, "forced"), env, t, T, history, ctx, 0))

    banner("THE SAME ROUND UNDER MAPPING 1  (answer letters exchanged)")
    print("Only the two option lines change. Probe 2 averages the A/B logit")
    print("difference over both mappings, cancelling any preference the model")
    print("has for a particular letter.\n")
    swapped = render(load_template(env_name, "forced"), env, t, T, history, ctx, 1)
    for line in swapped.splitlines():
        if line.strip().startswith(("A =", "B =")):
            print(line)

    banner("PROMPT SEEN BY PROBE 3  (chain of thought)")
    print("Identical body; only the closing instruction differs.\n")
    cot = render(load_template(env_name, "cot"), env, t, T, history, ctx, 0)
    body_end = cot.find("Explain your reasoning step by step:")
    print("[... same body as above ...]\n")
    print(cot[body_end:])

    banner("PROBE 3 FOR THE TWO REASONING MODELS")
    print("The single point at which any model receives different text: the")
    print("{answer_format} slot in step 4 becomes the phrasing recommended by")
    print("the DeepSeek-R1-Distill model cards.\n")
    boxed = render(load_template(env_name, "cot"), env, t, T, history, ctx, 0,
                   answer_format=ANSWER_FORMAT_BOXED)
    print(boxed[boxed.find("  4."):])


if __name__ == "__main__":
    main()
