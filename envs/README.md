# `envs/` — the Bayesian optimal baseline

Each file defines one scenario: a generative process **and** the exact posterior
inference over its own hidden parameter. This is the reference the paper
measures models against, so it is the part most worth auditing.

## The contract

Every scenario subclasses `SequentialEnv` (`base.py`) and provides:

| Method | Meaning |
|---|---|
| `new_round(t)` | per-round prompt variables (only Monty Hall uses this) |
| `behavior_action(t)` | what the **scripted** policy does this round — never the model |
| `outcome(t, ctx, a)` | sample an outcome under the hidden parameter; return a record |
| `format_history(records)` | render the history block shown in the prompt |
| `posterior(records)` | `p(θ | h_t)` over the hidden types |
| `oracle_p_act1(records)` | the baseline's score for action 1, in `[0,1]` |
| `hidden_param()` | the true hidden parameter, for logging only |

Two invariants make the comparison across models meaningful:

1. **The model never plays into the history.** Evidence is produced by
   `behavior_action`, so all models see identical histories per seed. The model
   is asked, as an observer, what the optimal action is *now*.
2. **The hidden parameter is fixed for the whole session** and drawn from a
   known uniform prior, so the posterior is exactly computable.

Actions are always named so that `act0` is the conservative option and `act1`
the evidence-sensitive one.

## The eight scenarios

**Bayesian inference family** — `θ` is a property of a non-strategic
environment; the observation model is a fixed stochastic rule.

| File | Hidden parameter | Decision rule |
|---|---|---|
| `bandit.py` | which arm is better (0.7 vs 0.3) | posterior P(arm better) > 1/2 |
| `monty.py` | placement rule ∈ {random, sticky, shifting} | posterior P(switch wins) > 1/2 |
| `trust_game.py` | repayment rate q ∈ {≈1, .8, .2, ≈0} | E[q \| h] > 2/3 |
| `machine.py` | payout rate q, same four values | E[q \| h] > 2/3 |

`machine` is `trust_game`'s de-socialised twin: identical hidden types,
utilities and — by seed pairing — identical outcome trajectories. It isolates
social framing while holding the inference problem fixed.

**Game-theoretic family** — `θ` is a stationary property of an opponent and the
utility comes from a payoff matrix. Because the opponent is a fixed policy drawn
from a known prior, these remain Bayesian decision problems, not equilibrium
computations.

| File | Hidden parameter | Threshold |
|---|---|---|
| `stag_hunt.py` | {always stag, always hare, tit-for-tat} | P(opponent stags) > 2/3 |
| `bos.py` | {always Bach, always Stravinsky, tit-for-tat} | P(opponent Bach) > 1/3 |
| `pennies.py` | {always heads, always tails, counter} | P(opponent heads) > 1/2 |
| `chicken.py` | yield rate q ∈ {.9, .7, .3, .1} | E[q \| h] > 1/2 |

`stag_hunt`, `bos` and `pennies` contain a **reactive** type whose move depends
on the agent's previous action; `chicken`'s opponent is non-reactive. That split
is what the paper's steering analysis turns on, and it is exposed as
`envs.REACTIVE`.

## One deliberate approximation

Every scenario except `bandit` computes an exact discrete posterior over a
finite type set. `bandit` instead scores arms with independent Beta(1,1)
posteriors over continuous rates, while its generative model uses a two-point
prior. The two are not the same posterior.

`demo/02_verify_oracle.py` quantifies this: across 36,000 decisions the two
select the same action 93.3% of the time, and **on the unambiguous steps the
paper actually scores, they agree 100%** — every disagreement falls inside the
near-indifference band that is excluded by design.

## Auditing this directory

```bash
python demo/02_verify_oracle.py      # posterior vs empirical frequency, all 8
python demo/03_belief_trace.py bos 2 # one session, round by round
```

`demo/02` never calls `posterior()` to produce its empirical numbers — it runs
the generative code, buckets sessions by the history they produced, and measures
how often each hidden type was the true one. Bayes says those frequencies must
equal `posterior()`. They do, to within sampling error.
