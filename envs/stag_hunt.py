"""Iterated Stag Hunt with opponent-type uncertainty.

Payoffs (row = you):
                 opp STAG   opp HARE
    you STAG        4          0
    you HARE        3          2

Hidden parameter: opponent type, uniform over
    ALWAYS_STAG : plays STAG w.p. 1-eps
    ALWAYS_HARE : plays HARE w.p. 1-eps
    TIT_FOR_TAT : copies YOUR previous move (STAG on round 1), w.p. 1-eps
eps = 0.10 execution noise keeps every observation informative but non-degenerate,
so the posterior is smooth rather than collapsing to a point mass.

Optimal action: EU(STAG)=4p, EU(HARE)=2+p where p = P(opp plays STAG next).
STAG is optimal iff p > 2/3. Oracle reports a smooth P(STAG optimal) via the
posterior-weighted probability that p exceeds 2/3 under each type (types are
point hypotheses, so this reduces to summing posterior mass of types whose
predicted p_stag > 2/3, with the mixed boundary handled by the noise).
"""

import numpy as np

from .base import SequentialEnv

TYPES = ("ALWAYS_STAG", "ALWAYS_HARE", "TIT_FOR_TAT")
EPS = 0.10


class StagHuntEnv(SequentialEnv):
    name = "stag_hunt"
    act0, act1 = "HARE", "STAG"
    act0_desc = "hunt HARE (safe: 2-3 points regardless of the other hunter)"
    act1_desc = "hunt STAG (4 points if the other hunter also hunts stag, 0 otherwise)"

    def __init__(self, rng, T, opp_type=None):
        super().__init__(rng, T)
        self.opp_type = opp_type if opp_type is not None else str(rng.choice(TYPES))

    def hidden_param(self):
        return f"opponent={self.opp_type}"

    def _p_opp_stag(self, opp_type, my_prev):
        if opp_type == "ALWAYS_STAG":
            return 1 - EPS
        if opp_type == "ALWAYS_HARE":
            return EPS
        # TIT_FOR_TAT
        if my_prev is None or my_prev == "STAG":
            return 1 - EPS
        return EPS

    def new_round(self, t):
        return {}

    def behavior_action(self, t):
        # random 50/50 scripted history player -- necessary to disambiguate
        # TIT_FOR_TAT from the two fixed types
        return int(self.rng.integers(0, 2))

    def outcome(self, t, ctx, action_idx):
        my = self.act1 if action_idx == 1 else self.act0
        my_prev = self._records[-1]["my"] if getattr(self, "_records", None) else None
        p_stag = self._p_opp_stag(self.opp_type, my_prev)
        opp = "STAG" if self.rng.random() < p_stag else "HARE"
        payoff = {("STAG", "STAG"): 4, ("STAG", "HARE"): 0,
                  ("HARE", "STAG"): 3, ("HARE", "HARE"): 2}[(my, opp)]
        record = {"t": t, "action": my, "my": my, "opp": opp,
                  "win": payoff >= 2, "payoff": payoff,
                  "result": f"opp played {opp}, you scored {payoff}"}
        self._records = getattr(self, "_records", []) + [record]
        return float(payoff), record

    def format_history(self, records):
        if not records:
            return "[]  # no completed rounds yet"
        lines = ["["]
        for r in records:
            lines.append(f'    {{"you": "{r["my"]}", "opponent": "{r["opp"]}", "your_payoff": {r["payoff"]}}},')
        lines.append("]")
        opp_stag = sum(1 for r in records if r["opp"] == "STAG")
        lines.append(f"# Tally  ->  opponent played STAG {opp_stag}/{len(records)} rounds")
        return "\n".join(lines)

    def posterior(self, records):
        logp = np.zeros(len(TYPES))
        my_prev = None
        for r in records:
            for i, ty in enumerate(TYPES):
                p = self._p_opp_stag(ty, my_prev)
                lik = p if r["opp"] == "STAG" else 1 - p
                logp[i] += np.log(max(lik, 1e-12))
            my_prev = r["my"]
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        my_prev = records[-1]["my"] if records else None
        p_stag_next = sum(post[i] * self._p_opp_stag(ty, my_prev)
                          for i, ty in enumerate(TYPES))
        # smooth decision score around the p > 2/3 threshold
        return float(1.0 / (1.0 + np.exp(-20 * (p_stag_next - 2 / 3))))
