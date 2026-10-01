"""Session-level metrics.

Semantic sign convention: every environment declares two semantic actions,
act0 (the 'conservative' option) and act1 (the 'informative' option, e.g. SWITCH,
STAG, COOPERATE, INVEST, RIGHT-arm). All gaps below live in semantic space:

    gap_sem > 0   model prefers act1
    gap_sem < 0   model prefers act0

Core quantities per round t:
    delta_pre[t]   semantic pre-CoT forced-choice gap (label-swap averaged)
    delta_post[t]  semantic post-CoT answer-token gap (None when format-invalid)
    delta_amp[t]   delta_post - delta_pre        (None when post invalid)
    oracle_p[t]    normative P(act1 optimal | h_t) from the environment's
                   exact/near-exact Bayesian oracle
Session summaries:
    drift_pre      correlation failure between oracle trajectory and pre-CoT
                   preference trajectory (Spearman)
    Delta_T        fraction of wrong decisions on unambiguous rounds (behavioral,
                   from the parsed CoT answer)
    format_rate    fraction of rounds with a valid generated answer token
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_session(rounds, ambiguity_eps=0.10):
    """rounds: list of per-round dicts produced by run_all.py."""
    T = len(rounds)
    oracle = [r["oracle_p_act1"] for r in rounds]
    pre = [r["delta_pre_sem"] for r in rounds]
    post = [r["delta_post_sem"] for r in rounds]
    valid = [r["post_valid"] for r in rounds]

    # behavioral error on unambiguous rounds
    wrong = 0
    n_unamb = 0
    for r in rounds:
        p = r["oracle_p_act1"]
        if abs(p - 0.5) <= ambiguity_eps:
            continue
        n_unamb += 1
        gt_act1 = p > 0.5
        if r["decision_act1"] is None:
            wrong += 1  # no decision on an unambiguous round counts as wrong
        elif r["decision_act1"] != gt_act1:
            wrong += 1
    Delta_T = wrong / n_unamb if n_unamb else None

    # drift: does the pre-CoT preference track the normative trajectory?
    drift_pre = spearman(oracle, pre)
    post_pairs = [(o, p) for o, p, v in zip(oracle, post, valid) if v]
    drift_post = spearman(*zip(*post_pairs)) if len(post_pairs) >= 3 else None

    amp = [p - q for p, q, v in zip(post, pre, valid) if v and p is not None]

    return {
        "T": T,
        "Delta_T": Delta_T,
        "n_unambiguous": n_unamb,
        "format_rate": sum(valid) / T,
        "drift_corr_pre": drift_pre,
        "drift_corr_post": drift_post,
        "mean_delta_pre": float(np.mean(pre)),
        "mean_abs_delta_pre": float(np.mean(np.abs(pre))),
        "mean_delta_amp": float(np.mean(amp)) if amp else None,
        "oracle_range": float(max(oracle) - min(oracle)),
    }
