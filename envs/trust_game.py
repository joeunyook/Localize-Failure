"""Repeated Trust Game (investor side) with trustee-type uncertainty.

Each round you hold 10 points. INVEST sends all 10, the transfer is tripled to
30, and the trustee returns some share; KEEP simply banks the 10.

Hidden parameter: trustee type, uniform over FOUR types (v2, 2026-07-19),
each defined by its per-round probability q of returning 15 (else 0):
    RECIPROCATOR : q ~ 1   returns 15 every time      (you net +5)
    GENEROUS     : q = 0.8 returns 15 about 4 in 5    (you net +2 in expectation)
    STINGY       : q = 0.2 returns 15 about 1 in 5    (you net -7 in expectation)
    EXPLOITER    : q ~ 0   returns 0 every time       (you net -10)

Why four types (v2 rebalance): the v1 design {RECIPROCATOR, EXPLOITER, NOISY}
mapped two of three types to KEEP, giving a 75/25 oracle-label base rate from
t=5 on -- any constant-sign responder scored 75% raw agreement. With two types
on each side of the EU threshold the asymptotic label split is ~50/50
(simulated census: majority class 51-59% at every t>=2, vs 59-75% in v1), and
the RECIPROCATOR-vs-GENEROUS discrimination makes the evidence genuinely
gradual instead of NOISY's instant identification on the first mixed outcome.

Oracle: exact posterior over the four types from observed returns (Bernoulli
likelihoods q^repaid * (1-q)^unrepaid; deterministic types epsilon-floored at
0.999/0.001 to avoid degeneracy).
    EU(INVEST | h) = 15 * sum_type P(type | h) * q_type
    INVEST optimal iff EU > 10. P(INVEST optimal) reported as a smooth sigmoid
    of (EU - 10) so trajectories are comparable across environments.
Prior EU = 15 * mean(q) = 7.5 < 10, unchanged from v1: the correct t=1 answer
is still KEEP (oracle sigmoid(-2.5) ~ 0.076) -- the optimism-bias diagnostic.

Implementation note: TYPES / RETURN_P / OUTCOME_KEY are CLASS attributes and
all methods dispatch through self, so envs/machine.py (the de-socialized twin)
can inherit the entire posterior/oracle/stratification machinery by overriding
only the names and the narrative wording. The twin relies on the type tuples
staying index-aligned (same q at the same position) and on this class not
consuming session rng in __init__ when seed is given -- that is what makes
seed s produce the SAME hidden type and the SAME payout sequence in both envs
(the paired-seed design).
"""

import numpy as np

from .base import SequentialEnv


class TrustGameEnv(SequentialEnv):
    name = "trust_game"
    act0, act1 = "KEEP", "INVEST"
    act0_desc = "KEEP your 10 points (safe, guaranteed 10)"
    act1_desc = "INVEST all 10 points (tripled to 30; the trustee decides how much to return)"

    # per-round P(outcome 15); also the likelihood model (epsilon-floored ends)
    TYPES = ("RECIPROCATOR", "GENEROUS", "STINGY", "EXPLOITER")
    RETURN_P = {"RECIPROCATOR": 0.999, "GENEROUS": 0.8,
                "STINGY": 0.2, "EXPLOITER": 0.001}
    OUTCOME_KEY = "returned"        # record field holding the 15/0 observation

    # the type draw is STRATIFIED across session seeds: every contiguous block
    # of 4 seeds contains each type exactly once (shuffled within the block).
    # An iid uniform draw at n=100 seeds landed 32/29/23/16 -- a ~2-sigma
    # fluctuation that reintroduces exactly the label imbalance the 4-type
    # design removes. Stratification makes the census exactly 25/25/25/25 over
    # seeds 0-99 (and 10 each over the steering seeds 0-39) while remaining
    # indistinguishable from a uniform draw within any single session.
    stratify_types = True

    def __init__(self, rng, T, trustee_type=None, seed=None):
        super().__init__(rng, T)
        if trustee_type is not None:
            self.trustee_type = trustee_type
        elif seed is not None:
            block, idx = divmod(int(seed), len(self.TYPES))
            perm = np.random.default_rng(block * 7919 + 12345).permutation(len(self.TYPES))
            self.trustee_type = self.TYPES[perm[idx]]
        else:
            self.trustee_type = str(rng.choice(self.TYPES))

    def hidden_param(self):
        return f"trustee={self.trustee_type}"

    def new_round(self, t):
        return {}

    def behavior_action(self, t):
        # history player always takes the informative action: outcomes are only
        # observable when the transfer/play happens, maximizing evidence
        return 1

    def _returned(self):
        # deterministic ends sample deterministically (the prompt's "every
        # time" is true in the data); the 0.999/0.001 epsilon lives only in
        # the likelihood model (anti-degeneracy). q-based so subclasses with
        # renamed types inherit identical rng behavior (pairing requirement).
        q = self.RETURN_P[self.trustee_type]
        if q >= 0.999:
            return 15
        if q <= 0.001:
            return 0
        return 15 if self.rng.random() < q else 0

    def outcome(self, t, ctx, action_idx):
        if action_idx == 0:
            record = {"t": t, "action": "KEEP", "returned": None,
                      "win": True, "payoff": 10, "result": "kept 10"}
            return 10.0, record
        ret = self._returned()
        record = {"t": t, "action": "INVEST", "returned": ret,
                  "win": ret > 10, "payoff": ret,
                  "result": f"trustee returned {ret}"}
        return float(ret), record

    def format_history(self, records):
        if not records:
            return "[]  # no completed rounds yet"
        lines = ["["]
        for r in records:
            if r["action"] == "KEEP":
                lines.append(f'    {{"you": "KEEP", "your_payoff": 10}},')
            else:
                lines.append(f'    {{"you": "INVEST", "trustee_returned": {r["returned"]}, "your_payoff": {r["payoff"]}}},')
        lines.append("]")
        inv = [r for r in records if r["action"] == "INVEST"]
        good = sum(1 for r in inv if r["returned"] == 15)
        lines.append(f"# Tally  ->  {good}/{len(inv)} investments were repaid with 15")
        return "\n".join(lines)

    def posterior(self, records):
        logp = np.zeros(len(self.TYPES))
        for r in records:
            if r["action"] != self.act1:
                continue
            paid15 = r[self.OUTCOME_KEY] == 15
            for i, ty in enumerate(self.TYPES):
                q = self.RETURN_P[ty]
                logp[i] += np.log(q if paid15 else 1.0 - q)
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        eu = 15.0 * float(sum(p * self.RETURN_P[ty]
                              for p, ty in zip(post, self.TYPES)))
        return float(1.0 / (1.0 + np.exp(-1.0 * (eu - 10))))


# module-level aliases kept for external scripts
TYPES = TrustGameEnv.TYPES
RETURN_P = TrustGameEnv.RETURN_P
