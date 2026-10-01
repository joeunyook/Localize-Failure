"""Iterated Matching Pennies with opponent-type uncertainty.

The suite's only STRICTLY OPPOSED payoff structure: there is no cell in which
both players do well. You score iff you match the opponent; the opponent's
interest is to mismatch. (As everywhere in this suite the opponent is a scripted
automaton whose own payoffs never enter the decision, so "zero-sum" describes
the payoff matrix shown to the model, not a strategic interaction -- the same
caveat bos carries for "conflicting preferences".)

Payoffs (row = you):
                 opp HEADS   opp TAILS
    you HEADS        1           0
    you TAILS        0           1

Hidden parameter: opponent type, uniform over
    ALWAYS_HEADS : plays HEADS w.p. 1-eps
    ALWAYS_TAILS : plays TAILS w.p. 1-eps
    COUNTER      : plays the OPPOSITE of your previous move w.p. 1-eps
                   (uniform on round 1, where there is no previous move)
eps = 0.10 execution noise keeps every observation informative and the posterior
smooth instead of collapsing to a point mass on the first move.

COUNTER is the natural reactive type in a strictly competitive game, and -- as
with TIT_FOR_TAT in stag_hunt/bos -- it is identifiable ONLY because the scripted
behavior policy plays both actions: its move correlates with your previous move
while the two fixed types ignore it entirely.

Optimal action: with p = P(opp plays HEADS next), EU(HEADS) = p and
EU(TAILS) = 1 - p, so HEADS is optimal iff p > 1/2. The EU gap is 2p-1,
identical in form to chicken's, so chicken's sigmoid sharpness (5) transfers.

NO DOMINANT ACTION: the best response flips with the posterior predictive, so
the optimal action is decided WITHIN the round and cannot depend on the
remaining horizon. This is the structural property that ipd lacks and that
makes this env safe to score against a stationary Eq.(2)-style baseline.

Round 1 is exactly indifferent by construction: the prior predictive is
(0.9 + 0.1 + 0.5)/3 = 0.5, so t=1 is dropped by the ambiguity filter -- the same
property chicken has.
"""

import numpy as np

from .base import SequentialEnv

TYPES = ("ALWAYS_HEADS", "ALWAYS_TAILS", "COUNTER")
EPS = 0.10
THRESHOLD = 0.5
SLOPE = 5


class PenniesEnv(SequentialEnv):
    name = "pennies"
    act0, act1 = "TAILS", "HEADS"
    act0_desc = "choose TAILS (1 point if they also choose Tails, 0 otherwise)"
    act1_desc = "choose HEADS (1 point if they also choose Heads, 0 otherwise)"

    def __init__(self, rng, T, opp_type=None):
        super().__init__(rng, T)
        self.opp_type = opp_type if opp_type is not None else str(rng.choice(TYPES))

    def hidden_param(self):
        return f"opponent={self.opp_type}"

    def _p_opp_heads(self, opp_type, my_prev):
        if opp_type == "ALWAYS_HEADS":
            return 1 - EPS
        if opp_type == "ALWAYS_TAILS":
            return EPS
        # COUNTER: plays the opposite of your previous move
        if my_prev is None:
            return 0.5
        return EPS if my_prev == "HEADS" else 1 - EPS

    def new_round(self, t):
        return {}

    def behavior_action(self, t):
        # random 50/50 scripted history player -- REQUIRED: COUNTER is only
        # distinguishable from the fixed types if your own moves vary.
        return int(self.rng.integers(0, 2))

    def outcome(self, t, ctx, action_idx):
        my = self.act1 if action_idx == 1 else self.act0
        my_prev = self._records[-1]["my"] if getattr(self, "_records", None) else None
        p_heads = self._p_opp_heads(self.opp_type, my_prev)
        opp = "HEADS" if self.rng.random() < p_heads else "TAILS"
        payoff = 1 if my == opp else 0
        record = {"t": t, "action": my, "my": my, "opp": opp,
                  "win": payoff == 1, "payoff": payoff,
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
        opp_heads = sum(1 for r in records if r["opp"] == "HEADS")
        lines.append(f"# Tally  ->  opponent chose HEADS {opp_heads}/{len(records)} rounds")
        return "\n".join(lines)

    def posterior(self, records):
        logp = np.zeros(len(TYPES))
        my_prev = None
        for r in records:
            for i, ty in enumerate(TYPES):
                p = self._p_opp_heads(ty, my_prev)
                lik = p if r["opp"] == "HEADS" else 1 - p
                logp[i] += np.log(max(lik, 1e-12))
            my_prev = r["my"]
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        my_prev = records[-1]["my"] if records else None
        p_heads_next = sum(post[i] * self._p_opp_heads(ty, my_prev)
                           for i, ty in enumerate(TYPES))
        # smooth decision score around the p > 1/2 threshold
        return float(1.0 / (1.0 + np.exp(-SLOPE * 2.0 * (p_heads_next - THRESHOLD))))
