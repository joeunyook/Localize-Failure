# `sample_data/` — bundled results

Enough real data to recompute the paper's central result without weights,
activations, or a GPU.

## `responses/`

Per-round records, 2 models x 8 scenarios x 100 seeds = 1600 files.
Models: `llama` (Llama-3.2 3B) and `qwen27` (Qwen-3.6 27B), chosen as a small
and a mid-scale model so the belief-to-action gap is visible at both ends.

Each file has a `rounds` list; the fields used by the demos are:

| Field | Meaning |
|---|---|
| `oracle_p_act1` | the baseline's score for action 1 at this step |
| `delta_pre_sem` | Probe 2: label-swap-averaged pre-CoT logit difference |
| `delta_post_sem` | Probe 3: the same quantity after chain-of-thought |
| `cot_text` | the model's generated reasoning, where collected |

`delta_pre_sem` is already averaged over the two answer-letter mappings, so its
sign is the model's semantic action preference with letter bias removed.

## `probes/`

Trained Probe 1 outputs for the same combinations, as
`{tag}__probe_{model}_{env}.json`. The field the demos read is
`layer_accuracy_bal`, the class-balanced held-out accuracy at each layer;
Belief (P1) is its value at the final layer.

Two tags appear because balanced accuracy was added after the first probe run;
`P1_v4_layerbal` carries it for combinations whose `P1_v4` file predates that
field. `demo/04` prefers `P1_v4_layerbal` and falls back.

## What is not here

Residual-stream activations (~74 GB) — needed to *re-train* Probe 1 or to run
the posterior-decodability analysis. The trained probes above are what they
produce. Also absent: the other seven models, the steering replays, and the
ablation sweeps.
