"""Chicken (Hawk-Dove) with opponent-type uncertainty (added 2026-07-19).

Payoffs (row = you):
                 opp YIELD   opp DARE
    you DARE        4           0
    you YIELD       3           1

Anti-coordination check: best response to YIELD is DARE (4 > 3), best
response to DARE is YIELD (1 > 0) -- the strategic-substitutes complement to
stag_hunt's coordination and ipd's dominant defection. Its role in the suite:
the existing social envs are all cooperation-flavored (a natural objection:
"RLHF prosociality explains the strategic failures"), and all have nontrivial
EU thresholds. Chicken is social with the TRIVIAL threshold: DARE is optimal
iff P(opp yields) > 1/2, so belief-sign equals action-sign, like bandit and
monty. Together with envs/machine.py (nonsocial, threshold 2/3) this fully
crosses {social x threshold}.

Hidden parameter: opponent type, one of FOUR yield rates (two per side of
the 1/2 threshold, symmetric -- the trust_game-v2 label-balance lesson):
    CAUTIOUS : yields w.p. 0.9
    WARY     : yields w.p. 0.7
    BOLD     : yields w.p. 0.3
    RECKLESS : yields w.p. 0.1
Moves are iid given the type (no reactive/TFT type, no horizon dependence,
hence no ipd-style degenerate endpoint labels). No separate execution-noise
line: the rates ARE the stochasticity, fully disclosed in the prompt.

Oracle: exact Bernoulli posterior over the four rates from the opponent's
observed moves (own moves are uninformative but are scripted 50/50 for
interface uniformity with stag/ipd). p = posterior-predictive P(yield next);
EU(DARE) = 4p, EU(YIELD) = 1 + 2p; oracle_p_act1 = sigmoid(5 * (2p - 1)).
Prior p = mean(rates) = 0.5 exactly, so t=1 is oracle 0.5 -- ambiguous by
construction (the uninformed-baseline round, like bandit t=1). Identified
types map to oracle ~0.98 / 0.88 / 0.12 / 0.02. Census (100 stratified
seeds): majority label 50-58% per t, P(correct side) climbs 0.50 -> 0.93
over the session -- the most gradual accumulation profile in the suite.
"""

import numpy as np

from .base import SequentialEnv

TYPES = ("CAUTIOUS", "WARY", "BOLD", "RECKLESS")
YIELD_P = {"CAUTIOUS": 0.9, "WARY": 0.7, "BOLD": 0.3, "RECKLESS": 0.1}
PAYOFF = {("DARE", "YIELD"): 4, ("DARE", "DARE"): 0,
          ("YIELD", "YIELD"): 3, ("YIELD", "DARE"): 1}
SLOPE = 5.0


class ChickenEnv(SequentialEnv):
    name = "chicken"
    act0, act1 = "YIELD", "DARE"
    act0_desc = "YIELD (safe: 3 if they yield, 1 if they dare)"
    act1_desc = "DARE (risky: 4 if they yield, 0 if they dare)"

    # stratified type draw across seeds (blocks of 4, shuffled per block),
    # same rationale as trust_game v2; distinct salt so chicken's block
    # shuffles are independent of the trust/machine ones
    stratify_types = True

    def __init__(self, rng, T, opp_type=None, seed=None):
        super().__init__(rng, T)
        if opp_type is not None:
            self.opp_type = opp_type
        elif seed is not None:
            block, idx = divmod(int(seed), len(TYPES))
            perm = np.random.default_rng(block * 7919 + 24601).permutation(len(TYPES))
            self.opp_type = TYPES[perm[idx]]
        else:
            self.opp_type = str(rng.choice(TYPES))

    def hidden_param(self):
        return f"opponent={self.opp_type}"

    def new_round(self, t):
        return {}

    def behavior_action(self, t):
        # 50/50 scripted history player: own moves carry no information about
        # a fixed-rate opponent, but the mixed history keeps the record format
        # and any future reactive-type extension parallel to stag/ipd
        return int(self.rng.integers(0, 2))

    def outcome(self, t, ctx, action_idx):
        my = self.act1 if action_idx == 1 else self.act0
        opp = "YIELD" if self.rng.random() < YIELD_P[self.opp_type] else "DARE"
        payoff = PAYOFF[(my, opp)]
        record = {"t": t, "action": my, "my": my, "opp": opp,
                  "win": payoff >= 3, "payoff": payoff,
                  "result": f"opp played {opp}, you scored {payoff}"}
        return float(payoff), record

    def format_history(self, records):
        if not records:
            return "[]  # no completed rounds yet"
        lines = ["["]
        for r in records:
            lines.append(f'    {{"you": "{r["my"]}", "opponent": "{r["opp"]}", "your_payoff": {r["payoff"]}}},')
        lines.append("]")
        opp_yield = sum(1 for r in records if r["opp"] == "YIELD")
        lines.append(f"# Tally  ->  opponent played YIELD {opp_yield}/{len(records)} rounds")
        return "\n".join(lines)

    def posterior(self, records):
        logp = np.zeros(len(TYPES))
        for r in records:
            yielded = r["opp"] == "YIELD"
            for i, ty in enumerate(TYPES):
                q = YIELD_P[ty]
                logp[i] += np.log(q if yielded else 1.0 - q)
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        p_yield = float(sum(p * YIELD_P[ty] for p, ty in zip(post, TYPES)))
        # EU(DARE) - EU(YIELD) = 4p - (1 + 2p) = 2p - 1; DARE optimal iff p > 1/2
        return float(1.0 / (1.0 + np.exp(-SLOPE * (2.0 * p_yield - 1.0))))
