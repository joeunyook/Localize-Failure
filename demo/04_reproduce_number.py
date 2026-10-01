#!/usr/bin/env python3
"""DEMO 4 - Reproduce the paper's central result from bundled data.
(~10 seconds, no GPU, no weights)

The paper's thesis in one line: models represent the decision-relevant belief
almost perfectly, yet fail to act on it -- and the failure is far worse under
game-theoretic structure. That is the gap between the Belief (P1) and
Pre-CoT (P2) rows of the main results table.

This script recomputes that gap for two models across all eight scenarios,
using only files in sample_data/:

    Pre-CoT (P2)  recomputed here from raw per-round records
                  (sample_data/responses/), exactly as in the paper
    Belief (P1)   read from the trained probe outputs
                  (sample_data/probes/); re-training it needs the residual
                  activations, which are far too large to ship

Both are class-balanced and computed only on unambiguous steps.

    python demo/04_reproduce_number.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "sample_data")

AMBIG = 0.10
MODELS = [("llama", "Llama-3.2 3B"), ("qwen27", "Qwen-3.6 27B")]
BAYESIAN = ["bandit", "monty", "trust_game", "machine"]
GAME = ["stag_hunt", "bos", "pennies", "chicken"]
ENVS = BAYESIAN + GAME

# Values printed in the paper's main results table, for comparison.
PAPER = {
    ("llama", "bandit"): (98, 71), ("llama", "monty"): (98, 68),
    ("llama", "trust_game"): (100, 73), ("llama", "machine"): (100, 41),
    ("llama", "stag_hunt"): (87, 50), ("llama", "bos"): (87, 50),
    ("llama", "pennies"): (87, 50), ("llama", "chicken"): (98, 53),
    ("qwen27", "bandit"): (100, 95), ("qwen27", "monty"): (100, 99),
    ("qwen27", "trust_game"): (100, 88), ("qwen27", "machine"): (100, 84),
    ("qwen27", "stag_hunt"): (92, 71), ("qwen27", "bos"): (92, 82),
    ("qwen27", "pennies"): (92, 75), ("qwen27", "chicken"): (99, 82),
}


def balanced(pairs):
    """Macro-average accuracy over the two optimal-action classes (chance 50)."""
    cls = defaultdict(lambda: [0, 0])
    for gt, pred in pairs:
        c = cls[bool(gt)]
        c[1] += 1
        if pred is not None and bool(pred) == bool(gt):
            c[0] += 1
    if len(cls) < 2:
        return None
    return 100.0 * sum(h / n for h, n in cls.values()) / len(cls)


def pre_cot(model, env):
    """P2: balanced agreement between the sign of the pre-CoT preference and
    the baseline's optimal action, on unambiguous steps only."""
    pairs = []
    for path in glob.glob(os.path.join(DATA, "responses",
                                       f"{model}_{env}_seed*.json")):
        with open(path) as fh:
            rounds = json.load(fh).get("rounds", [])
        for r in rounds:
            p = r.get("oracle_p_act1")
            if p is None or abs(p - 0.5) <= AMBIG:
                continue
            g = r.get("delta_pre_sem")
            pairs.append((p > 0.5, None if g is None else g > 0))
    return (balanced(pairs), len(pairs))


def belief(model, env):
    """P1: class-balanced probe accuracy at the final layer."""
    for tag in ("P1_v4_layerbal", "P1_v4"):
        path = os.path.join(DATA, "probes", f"{tag}__probe_{model}_{env}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            lab = json.load(fh).get("layer_accuracy_bal")
        if lab:
            return 100.0 * lab[str(max(int(k) for k in lab))]
    return None


def main():
    print("=" * 78)
    print("REPRODUCING THE BELIEF-TO-ACTION GAP FROM BUNDLED DATA")
    print("=" * 78)
    print("P2 is recomputed from raw per-round records; P1 is read from the")
    print("trained probes. 'paper' is the value printed in the main table.\n")

    for model, label in MODELS:
        print(f"--- {label} " + "-" * (74 - len(label)))
        print(f"{'scenario':<13}{'P1':>7}{'P2':>7}{'gap':>7}"
              f"{'  |':>3}{'paper P1':>10}{'paper P2':>10}{'  steps':>9}")
        gaps = {"bayes": [], "game": []}
        for env in ENVS:
            p1 = belief(model, env)
            p2, n = pre_cot(model, env)
            if p1 is None or p2 is None:
                print(f"{env:<13}{'  (missing from sample_data)':>40}")
                continue
            pp1, pp2 = PAPER.get((model, env), (None, None))
            bucket = "bayes" if env in BAYESIAN else "game"
            gaps[bucket].append(p1 - p2)
            print(f"{env:<13}{p1:>7.1f}{p2:>7.1f}{p1 - p2:>7.1f}"
                  f"{'  |':>3}{pp1:>10}{pp2:>10}{n:>9}")
        for k, name in (("bayes", "Bayesian inference"),
                        ("game", "game theoretic")):
            if gaps[k]:
                print(f"{'mean gap, ' + name:<27}{sum(gaps[k]) / len(gaps[k]):>7.1f}"
                      f"  percentage points")
        print()

    print("=" * 78)
    print("The P1 column sits near ceiling everywhere, including the scenarios")
    print("where P2 collapses to chance. The belief is present and linearly")
    print("decodable; what fails is the mapping from that belief to an action,")
    print("and it fails much harder under game-theoretic structure.")
    print("\nSmall differences from the paper column are expected: the table")
    print("rounds to whole percentage points, and sample_data/ ships two of the")
    print("nine models.")


if __name__ == "__main__":
    main()
