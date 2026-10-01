# Localizing LLM Failures in Decision Making with Bayesian Optimal Baselines

This repository contains the parts of the pipeline needed to understand and check the
paper: **the Bayesian baseline design**, **the three probing points**, **the
exact prompts**, and **runnable demonstrations that need no GPU and no model
weights**.

It is deliberately not the full research tree. Model weights (132 GB) and raw
residual-stream activations (~74 GB) cannot be shipped; everything that depends
on them is either precomputed and bundled, or clearly marked.

---

## Start here: a 30-minute path

| Minutes | Do this | Why |
|---|---|---|
| 0–5 | Read this file | What is measured and how |
| 5–10 | `python demo/01_render_prompt.py` | See the exact string a model receives |
| 10–15 | Read [`envs/README.md`](envs/README.md), skim `envs/bandit.py` then `envs/stag_hunt.py` | The baseline: prior, likelihood, utility, decision rule |
| 15–20 | `python demo/03_belief_trace.py` | Watch the baseline's belief update round by round |
| 20–25 | Read [`probes/README.md`](probes/README.md) | Where P1/P2/P3 read from, and why those points |
| 25–30 | `python demo/04_reproduce_number.py` | Reproduce the paper's central result from bundled data |

If you have only five minutes, run `demo/04_reproduce_number.py`. It recomputes
the Belief-vs-Pre-CoT gap — the paper's thesis — from raw per-round records.

If you want to check that the "Bayesian optimal baseline" really is optimal,
run `demo/02_verify_oracle.py` (~1–2 min). It validates each scenario's
analytic posterior against empirical frequencies from the scenario's own
generative code, so agreement is evidence rather than tautology.

---

## What is being measured

Each scenario is a repeated binary decision under a hidden parameter `θ` that is
fixed for a session and inferable from accumulating evidence. A scripted
behaviour policy — not the evaluated model — generates the history, so every
model sees an identical evidence stream for a given seed.

The baseline updates `p(θ | h_t)` by Bayes and picks the action maximising
posterior expected utility. Model behaviour is compared to it at three points:

| Probe | Reads | When |
|---|---|---|
| **P1 Belief** | residual stream, every layer, final prompt token | forward pass, no generation |
| **P2 Pre-CoT** | vocabulary logits for the two answer letters | same forward pass |
| **P3 Post-CoT** | the same logit difference at the committed answer | after the model generates its reasoning |

P1 and P2 read the *same* forward pass and differ only in what is read from it.
That is what makes "the belief is represented but not used" a measurable claim
rather than an interpretation.

---

## Layout

```
envs/         the Bayesian baseline: 8 scenarios, each with an exact posterior
probes/       the three probing points
prompts/      the exact prompt templates, 8 scenarios x {forced, cot}
prompting.py  prompt rendering + label-swap logic (torch-free)
demo/         four runnable scripts, no GPU, no weights
sample_data/  per-round records and trained probes for 2 of the 9 models
pipeline/     full experiment drivers (need GPU + weights)
```

---

## Two details that matter for reading the numbers

**Near-indifference exclusion.** Steps where the baseline is within `0.10` of
indifference (`|p*-0.5| <= 0.10`) are dropped from every reported metric. At
those steps the optimal action is decided by an arbitrarily small utility
difference, so the binary label carries no reliable signal. The same filter
applies to P1, P2 and P3, so the three are computed on identical step sets.

**Class-balanced metrics.** Several scenarios have skewed optimal-action base
rates, so a constant responder can score well above 50% on raw agreement. Every
number in the paper is macro-averaged over the two optimal-action classes, which
pins any constant predictor at exactly 50.

---

## Reproducibility boundary

Bundled, so reproducible here:

- all scenario definitions, oracles and prompts
- per-round records for 2 models × 8 scenarios × 100 seeds
- trained P1 probes for those combinations
- P2 recomputation from raw records (`demo/04`)
- oracle correctness (`demo/02`)

Not bundled, so **not** reproducible here:

- **model weights** — load from the identifiers in `pipeline/model_loader.py`
- **residual-stream activations** — needed to *re-train* P1 or to run the
  posterior-decodability analysis; the trained outputs are bundled instead
- the other 7 models' records, and the ablation sweeps

---

## Requirements

`demo/` needs only `numpy` (and `scikit-learn` for probe training).
`pipeline/` additionally needs `torch` and `transformers` plus GPU access.
See `requirements.txt`.

```bash
pip install -r requirements.txt
python demo/01_render_prompt.py
```
