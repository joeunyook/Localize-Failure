"""Iterated Bach-or-Stravinsky (Battle of the Sexes) with opponent-type uncertainty.

Structural twin of stag_hunt: a coordination game with NO dominant action, so the
posterior over the opponent's next move determines the optimal action WITHIN the
current round. Unlike ipd, nothing here depends on the remaining horizon.

Payoffs (row = you):
                 opp BACH   opp STRAV
    you BACH        2           0
    you STRAV       0           1

Both players want to coordinate, but on different venues: you prefer BACH (2),
they prefer STRAVINSKY. Mismatch pays 0 to you either way.

Hidden parameter: opponent type, uniform over
    ALWAYS_BACH  : plays BACH w.p. 1-eps   (accommodates your preference)
    ALWAYS_STRAV : plays STRAV w.p. 1-eps  (insists on their preference)
    TIT_FOR_TAT  : copies YOUR previous move (BACH on round 1), w.p. 1-eps
eps = 0.10 execution noise keeps every observation informative but non-degenerate,
so the posterior is smooth rather than collapsing to a point mass.

Optimal action: EU(BACH)=2p, EU(STRAV)=1-p where p = P(opp plays BACH next).
BACH is optimal iff 2p > 1-p, i.e. p > 1/3 -- a third threshold value alongside
stag_hunt's 2/3 and chicken's 1/2. The EU gap is 3p-1, whose slope in p matches
stag_hunt's 3p-2, so the same sigmoid sharpness (20) gives a comparable oracle.
"""

import numpy as np

from .base import SequentialEnv

TYPES = ("ALWAYS_BACH", "ALWAYS_STRAV", "TIT_FOR_TAT")
EPS = 0.10
THRESHOLD = 1.0 / 3.0
SLOPE = 20


class BoSEnv(SequentialEnv):
    name = "bos"
    act0, act1 = "STRAVINSKY", "BACH"
    act0_desc = "choose STRAVINSKY (1 point if they also choose Stravinsky, 0 otherwise)"
    act1_desc = "choose BACH (2 points if they also choose Bach, 0 otherwise)"

    def __init__(self, rng, T, opp_type=None):
        super().__init__(rng, T)
        self.opp_type = opp_type if opp_type is not None else str(rng.choice(TYPES))

    def hidden_param(self):
        return f"opponent={self.opp_type}"

    def _p_opp_bach(self, opp_type, my_prev):
        if opp_type == "ALWAYS_BACH":
            return 1 - EPS
        if opp_type == "ALWAYS_STRAV":
            return EPS
        # TIT_FOR_TAT
        if my_prev is None or my_prev == "BACH":
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
        p_bach = self._p_opp_bach(self.opp_type, my_prev)
        opp = "BACH" if self.rng.random() < p_bach else "STRAVINSKY"
        payoff = {("BACH", "BACH"): 2, ("BACH", "STRAVINSKY"): 0,
                  ("STRAVINSKY", "BACH"): 0, ("STRAVINSKY", "STRAVINSKY"): 1}[(my, opp)]
        record = {"t": t, "action": my, "my": my, "opp": opp,
                  "win": payoff >= 1, "payoff": payoff,
                  "result": f"opp chose {opp}, you scored {payoff}"}
        self._records = getattr(self, "_records", []) + [record]
        return float(payoff), record

    def format_history(self, records):
        if not records:
            return "[]  # no completed rounds yet"
        lines = ["["]
        for r in records:
            lines.append(f'    {{"you": "{r["my"]}", "opponent": "{r["opp"]}", "your_payoff": {r["payoff"]}}},')
        lines.append("]")
        opp_bach = sum(1 for r in records if r["opp"] == "BACH")
        lines.append(f"# Tally  ->  opponent chose BACH {opp_bach}/{len(records)} rounds")
        return "\n".join(lines)

    def posterior(self, records):
        logp = np.zeros(len(TYPES))
        my_prev = None
        for r in records:
            for i, ty in enumerate(TYPES):
                p = self._p_opp_bach(ty, my_prev)
                lik = p if r["opp"] == "BACH" else 1 - p
                logp[i] += np.log(max(lik, 1e-12))
            my_prev = r["my"]
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        my_prev = records[-1]["my"] if records else None
        p_bach_next = sum(post[i] * self._p_opp_bach(ty, my_prev)
                          for i, ty in enumerate(TYPES))
        # smooth decision score around the p > 1/3 threshold
        return float(1.0 / (1.0 + np.exp(-SLOPE * (p_bach_next - THRESHOLD))))
