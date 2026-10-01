"""Two-armed Bernoulli bandit (pure Bayesian evidence accumulation, no host).

Hidden parameter: which arm is the good one. (p_good, p_bad) = (0.7, 0.3),
assignment to LEFT/RIGHT randomized per session.

Oracle: independent Beta(1,1) posteriors per arm; P(RIGHT optimal) =
P(p_R > p_L | h_t) estimated by fixed-seed Monte Carlo over the two Betas
(20k samples; SE < 0.4%, far below the ambiguity threshold used downstream).
"""

import numpy as np

from .base import SequentialEnv, tally_block


class BanditEnv(SequentialEnv):
    name = "bandit"
    act0, act1 = "LEFT", "RIGHT"
    act0_desc = "pull the LEFT arm"
    act1_desc = "pull the RIGHT arm"

    def __init__(self, rng, T, p_good=0.7, p_bad=0.3):
        super().__init__(rng, T)
        right_is_good = bool(rng.integers(0, 2))
        self.p = {"LEFT": p_bad if right_is_good else p_good,
                  "RIGHT": p_good if right_is_good else p_bad}
        self._mc = np.random.default_rng(12345)  # fixed oracle MC stream

    def hidden_param(self):
        return f"p_LEFT={self.p['LEFT']}, p_RIGHT={self.p['RIGHT']}"

    def new_round(self, t):
        return {}

    def behavior_action(self, t):
        # alternate arms so both are sampled evenly regardless of seed
        return t % 2

    def outcome(self, t, ctx, action_idx):
        arm = self.act1 if action_idx == 1 else self.act0
        win = self.rng.random() < self.p[arm]
        record = {"t": t, "action": arm, "win": bool(win),
                  "result": "WON" if win else "LOST"}
        return (1.0 if win else 0.0), record

    def format_history(self, records):
        return tally_block(records, self.act0, self.act1)

    def oracle_p_act1(self, records):
        aL = 1 + sum(1 for r in records if r["action"] == "LEFT" and r["win"])
        bL = 1 + sum(1 for r in records if r["action"] == "LEFT" and not r["win"])
        aR = 1 + sum(1 for r in records if r["action"] == "RIGHT" and r["win"])
        bR = 1 + sum(1 for r in records if r["action"] == "RIGHT" and not r["win"])
        sL = self._mc.beta(aL, bL, 20000)
        sR = self._mc.beta(aR, bR, 20000)
        return float(np.mean(sR > sL))
