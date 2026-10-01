# `pipeline/` — full experiment drivers

These are the scripts that produced the results in the paper. They require GPU
access and model weights, so they will not run from this bundle as-is. They are
included so the measurement code can be read, not so it can be re-executed.

| File | Role |
|---|---|
| `run_all.py` | main sweep: renders prompts, runs the forward pass, reads Probes 1 and 2, optionally generates and reads Probe 3 |
| `run_steer.py` | causal intervention sweep (dose-response, anti and random controls) |
| `model_loader.py` | the nine-model registry and all per-model decoding configuration |
| `io_utils.py`, `metrics.py` | serialisation and metric helpers |

## Model weights are not included

`model_loader.py` lists the exact Hugging Face identifier for each of the nine
models. Load them from the hub, or edit the `id` field to a local path.

Three configuration details in that file are methodologically load-bearing:

1. **Decoding.** The two DeepSeek-R1-Distill models generate with
   `temperature=0.6`, `top_p=0.95`, as their model cards require to avoid
   degenerate repetition. All other models decode greedily. Sampling affects
   Probe 3 only; Probes 1 and 2 read a single forward pass and never sample.

2. **`prefill_prefix`.** The R1-Distill chat templates open a `<think>` block.
   For the forced-choice pass only, a closing `<think>\n\n</think>` prefill is
   appended after the chat template so the answer token is the model's next real
   token and the pre-CoT preference is measured at the intended position. Probe 3
   leaves reasoning enabled and applies no prefill.

3. **`tokenizer_patch_source`.** The published tokenizer for
   `DeepSeek-R1-Distill-Llama-70B` contains no pre-tokenizer, so whitespace is
   discarded at encode time and every prompt becomes a single run-on string.
   Because that model is a fine-tune of Llama-3.3-70B-Instruct with an unchanged
   vocabulary, the pre-tokenizer and decoder are transplanted from it at load
   time, preserving DeepSeek's chat template and `<think>` tokens. All reported
   results for that model use the corrected tokenizer.

## Scale of the reported runs

Probes 1 and 2: 100 seeds x 10 steps, 9 models x 8 scenarios.
Probe 3: the first 40 seeds, because it requires full generation.

## Note on `prompting.py`

`run_all.py` contains its own copy of the prompt-rendering logic, exactly as it
ran during the experiments. The top-level `prompting.py` is the same
substitution extracted without the torch dependency, so the `demo/` scripts run
on a laptop. They are kept separate so this directory stays a faithful record of
what was executed.
