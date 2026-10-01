"""Slot machine with payout-process uncertainty: the DE-SOCIALIZED TWIN of the
trust game (added 2026-07-19).

Identical inference problem and payoffs to trust_game v2 -- the agentic
trustee is replaced by a mechanical payout process, and every social word
(trust, return, decide, dispositions with trait names) is removed from the
prompt. Hidden parameter: process type, each defined by its per-round
probability q of paying out 15 (else 0):
    ALWAYS_PAY : q ~ 1
    HIGH_RATE  : q = 0.8
    LOW_RATE   : q = 0.2
    NEVER_PAY  : q ~ 0
PLAY costs the 10-point endowment and yields 15 or 0; PASS banks 10.
EU(PLAY) = 15 * posterior-mean q; PLAY optimal iff E[q] > 2/3. Prior EU = 7.5
< 10, so the correct t=1 answer is PASS (oracle sigmoid(-2.5) ~ 0.076) --
identical to trust_game by construction.

Why it exists: the working inference envs (bandit, monty) all have a trivial
"pick the likelier option" threshold of 0.5, and the failing social envs all
have a nontrivial EU threshold -- so "the decision circuit ignores the belief
in social games" is confounded with "models fail whenever the decision needs
an EU-threshold comparison". This env is nonsocial WITH the 2/3 threshold:
it isolates which of the two explanations holds.

Where the threshold actually bites (audited 2026-07-19): with these sharp
type rates the posterior-mean E[q] jumps to ~0.84 after one payout and ~0.16
after one miss, so the mid-band E[q] in (0.5, 2/3) is nearly empty (2/1000
rounds at 100 seeds) -- do NOT build a "band rounds" analysis. The
well-populated dissociation rounds are the NEUTRAL-BELIEF ones: t=1 and any
tied history (k paid, k unpaid) give E[q] = 0.5 exactly -- no likelier
hypothesis at all -- while the correct action is decisively PASS (oracle
0.076, unambiguous). A likelier-option responder has zero signal there; an
EU computer answers PASS. trust_game has the identical property (same math),
so the social/nonsocial pairing is preserved on exactly these rounds too.

PAIRED-SEED DESIGN: this class subclasses TrustGameEnv and overrides only
names and wording. TYPES is index-aligned with trust's (same q at the same
position), the stratified draw formula is inherited, and rng consumption is
identical -- so seed s produces the SAME hidden type (under the name mapping
RECIPROCATOR->ALWAYS_PAY, GENEROUS->HIGH_RATE, STINGY->LOW_RATE,
EXPLOITER->NEVER_PAY) and the SAME 15/0 outcome sequence as trust_game seed
s. Every round is a matched pair: identical evidence, identical oracle,
social vs mechanical framing. Verified by the smoke test.
"""

from .trust_game import TrustGameEnv


class MachineEnv(TrustGameEnv):
    name = "machine"
    act0, act1 = "PASS", "PLAY"
    act0_desc = "PASS and bank your 10 points (safe, guaranteed 10)"
    act1_desc = "PLAY the machine with your 10 points (pays out 15 or 0 according to its hidden process)"

    # index-aligned with TrustGameEnv.TYPES -- do not reorder (pairing breaks)
    TYPES = ("ALWAYS_PAY", "HIGH_RATE", "LOW_RATE", "NEVER_PAY")
    RETURN_P = {"ALWAYS_PAY": 0.999, "HIGH_RATE": 0.8,
                "LOW_RATE": 0.2, "NEVER_PAY": 0.001}
    OUTCOME_KEY = "paid"

    def hidden_param(self):
        return f"process={self.trustee_type}"

    def outcome(self, t, ctx, action_idx):
        if action_idx == 0:
            record = {"t": t, "action": "PASS", "paid": None,
                      "win": True, "payoff": 10, "result": "banked 10"}
            return 10.0, record
        paid = self._returned()
        record = {"t": t, "action": "PLAY", "paid": paid,
                  "win": paid > 10, "payoff": paid,
                  "result": f"machine paid out {paid}"}
        return float(paid), record

    def format_history(self, records):
        if not records:
            return "[]  # no completed rounds yet"
        lines = ["["]
        for r in records:
            if r["action"] == "PASS":
                lines.append(f'    {{"you": "PASS", "your_payoff": 10}},')
            else:
                lines.append(f'    {{"you": "PLAY", "machine_paid": {r["paid"]}, "your_payoff": {r["payoff"]}}},')
        lines.append("]")
        plays = [r for r in records if r["action"] == "PLAY"]
        good = sum(1 for r in plays if r["paid"] == 15)
        lines.append(f"# Tally  ->  {good}/{len(plays)} plays paid out 15")
        return "\n".join(lines)
