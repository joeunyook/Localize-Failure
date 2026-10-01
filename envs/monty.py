"""Hidden-rule iterated Monty Hall.

Hidden parameter: prize-placement rule, uniform over {random, sticky, shifting}.
    random   : prize uniform each round      -> P(SWITCH wins) = 2/3
    sticky   : prize behind initial pick     -> SWITCH always loses
    shifting : prize behind the switch door  -> SWITCH always wins

Exact Bayesian oracle over the three rules from (action, result) pairs:
    L(win | STAY,   random)=1/3   L(win | SWITCH, random)=2/3
    L(win | STAY,   sticky)=1     L(win | SWITCH, sticky)=0
    L(win | STAY,   shifting)=0   L(win | SWITCH, shifting)=1
"""

import numpy as np

from .base import SequentialEnv, tally_block

RULES = ("random", "sticky", "shifting")

WIN_LIK = {  # P(win | action, rule)
    ("STAY", "random"): 1 / 3, ("SWITCH", "random"): 2 / 3,
    ("STAY", "sticky"): 1.0,   ("SWITCH", "sticky"): 0.0,
    ("STAY", "shifting"): 0.0, ("SWITCH", "shifting"): 1.0,
}


class MontyEnv(SequentialEnv):
    name = "monty"
    act0, act1 = "STAY", "SWITCH"
    act0_desc = "keep your original door"
    act1_desc = "move to the remaining closed door"

    def __init__(self, rng, T, rule=None):
        super().__init__(rng, T)
        self.rule = rule if rule is not None else str(rng.choice(RULES))

    def hidden_param(self):
        return f"rule={self.rule}"

    def new_round(self, t):
        doors = [1, 2, 3]
        initial_pick = int(self.rng.choice(doors))
        if self.rule == "sticky":
            prize = initial_pick
        elif self.rule == "shifting":
            prize = None  # determined after host opens
        else:
            prize = int(self.rng.choice(doors))
        # host opens a goat door that is not the pick
        candidates = [d for d in doors if d != initial_pick and d != prize]
        host_opens = int(self.rng.choice(candidates))
        switch_door = [d for d in doors if d not in (initial_pick, host_opens)][0]
        if self.rule == "shifting":
            prize = switch_door
        return {
            "initial_pick": initial_pick,
            "host_opens": host_opens,
            "switch_door": switch_door,
            "prize": prize,
        }

    def behavior_action(self, t):
        # 50/50 scripted history player: both actions observed, rule identifiable
        return int(self.rng.integers(0, 2))

    def outcome(self, t, ctx, action_idx):
        action = self.act1 if action_idx == 1 else self.act0
        final_door = ctx["switch_door"] if action_idx == 1 else ctx["initial_pick"]
        win = final_door == ctx["prize"]
        reward = 1.0 if win else 0.0
        record = {"t": t, "action": action, "win": bool(win),
                  "result": "WON" if win else "LOST"}
        return reward, record

    def format_history(self, records):
        return tally_block(records, self.act0, self.act1)

    def posterior(self, records):
        logp = np.zeros(len(RULES))
        for r in records:
            for i, rule in enumerate(RULES):
                p = WIN_LIK[(r["action"], rule)]
                lik = p if r["win"] else 1.0 - p
                logp[i] += np.log(max(lik, 1e-12))
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        p_switch_wins = sum(post[i] * WIN_LIK[("SWITCH", rule)]
                            for i, rule in enumerate(RULES))
        return float(p_switch_wins)
