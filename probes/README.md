# `probes/` — the three probing points

The baseline solves each task in three stages: observe the history, update the
belief over the hidden parameter, map that belief to an action through the
expected-utility margin. The three probes give interpretable access to the
corresponding stages of the model's own process.

The key design point: **Probes 1 and 2 read the same forward pass.** The prompt
ends with an instruction not to explain, no tokens are generated, and both the
residual stream and the answer logits are read at the final prompt token. So
when P1 is at ceiling and P2 is at chance, that is not two experiments
disagreeing — it is one forward pass in which the information is present and
unused.

---

## Probe 1 — belief representation (`probe_train.py`)

**Target.** The baseline's decision state, `y = 1[ΔU_t > 0]`, where `ΔU_t` is the
posterior expected-utility margin between the two actions.

**Read point.** Residual stream at the final prompt token, every layer
`0..L` (layer 0 is the token embedding).

**Estimator.** ℓ2 logistic regression, `C = 0.1`, fitted independently per layer
on unstandardised features. `C` is fixed, not tuned. With `d_model >> n`, the
strong penalty is what makes the fit identifiable.

**Cross-validation.** Leave-one-seed-out: one fold per seed, all `T` steps of a
seed held out together. A seed fixes the hidden parameter and the whole evidence
trajectory, so this prevents scoring a probe on a history it partly trained on.

**Metric.** Class-balanced accuracy — the macro-average over the two
optimal-action classes of the pooled held-out predictions. Any constant
predictor scores exactly 50. The number reported as Belief (P1) is the balanced
accuracy at the **final layer**, which involves no layer selection and so
carries no selection optimism.

A single best layer `ℓ*` is also recorded, chosen on raw accuracy. It is used
**only** to site the causal intervention below — never to report a decoding
accuracy.

---

## Probe 2 — belief-to-action mapping (`readout.py`)

**Read point.** Vocabulary logits at the same final position of the same forward
pass. Logits for each answer letter are aggregated across tokenizer variants
with log-sum-exp; their difference is the model's action preference before it
has generated anything.

**Label-swap averaging.** Every prompt is rendered under both assignments of
semantic actions to answer letters, and the logit difference is averaged over
the two. This cancels any intrinsic preference for a particular letter, which is
otherwise large enough to dominate the measurement in small models — one model
in the study has a mean absolute letter bias four to ten times every other's.

**Metric.** Balanced agreement between the sign of that averaged difference and
the baseline's optimal action.

---

## Probe 3 — post-CoT commitment (`readout.py`)

The model generates its full reasoning response. We locate the step at which it
commits to a final answer and read the same logit difference there, from the
**raw** logits before sampling filters are applied — reading post-filter scores
silently corrupts the gap for any model decoded with top-p.

A response with no parseable answer yields no measurement, so this quantity is
defined only over responses that terminate. Non-termination is itself a
reported result rather than missing data.

Answers are parsed with the **last** matching answer pattern, not the first:
models frequently restate the prompt's format instruction before giving their
own answer, and first-match parsing silently locks onto the echoed instruction.

---

## Causal intervention (`steer.py`)

Probes are correlational. To test whether the decoded decision state is
*used*, the intervention adds it back to the residual stream and re-reads the
action logits.

At the probe's best layer `ℓ*`, with unit weight vector `ŵ` and
`σ_proj = sd(X ŵ)`:

```
h  <-  h  +  α · σ_proj · sign · v
```

where `sign` orients the push toward the baseline-optimal action, and `v` is
either `ŵ` (rescue), `-ŵ` (anti), or one of two norm-matched isotropic random
directions (control). Scaling by `σ_proj` expresses strength in units of the
model's own variation along that direction, so models at different
representation scales are comparable. The full dose–response `α ∈ {1,2,4,8}` is
evaluated rather than a single chosen strength, and all conditions for a step
run in one batched forward pass over the identical tokenisation.

The steering direction is refit on all data rather than cross-validated. That is
admissible because the claim is causal — does this intervention change the
action? — and not a generalisation claim about a classifier.

---

## What runs without weights

`probe_train.py` and `steer.py` need residual-stream activations and model
weights respectively, neither of which is shipped. Their outputs are bundled in
`sample_data/probes/`, and `demo/04_reproduce_number.py` recomputes Probe 2 in
full from raw per-round records.
