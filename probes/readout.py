"""The three measurement instruments.

  1. pre_cot_probe      : single forward pass on a forced-choice prompt that ends
                          exactly at 'Answer:'. Reads the A/B logit gap at the last
                          position BEFORE any generation. Also captures the residual
                          stream h_k[last_prompt_token] for every layer k, and the
                          layer-wise lens trajectory (norm+lm_head applied to each
                          layer's residual).
  2. post_cot_generate  : full CoT generation. Finds the generated 'Answer: A/B'
                          token and reads the logit gap AT that generation step.
                          NO fallback: if the answer token is never generated the
                          round is marked invalid and excluded from logit metrics.
  3. (analysis-time)    : linear probes are trained OFFLINE on the residuals saved
                          by pre_cot_probe -- see analysis/probe_train.py.

Sign convention: gap = logit(B-side tokens) - logit(A-side tokens), computed with
logsumexp over tokenizer variants. The runner converts label-space gaps into
SEMANTIC gaps (act1 minus act0) using the active label mapping, so label-swap
controls cancel any letter bias.
"""

import re
import torch


def _group_logit(logits_1d, ids):
    """logsumexp aggregate over token-id variants of one answer letter."""
    return torch.logsumexp(logits_1d[ids], dim=0).item()


def _find_norm(model):
    """Locate the final pre-unembed LayerNorm/RMSNorm across architectures."""
    for path in ("model.norm", "model.final_layernorm", "transformer.ln_f"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


@torch.no_grad()
def pre_cot_probe(model, tok, device, forced_prompt, ids_A, ids_B,
                  capture_residuals=True, capture_lens=True):
    """Forward pass only -- zero generated tokens.

    Returns dict with:
      gap_raw        : z_B - z_A at the final position (label space)
      logit_A/logit_B: the aggregated letter logits
      residuals      : (n_layers+1, d_model) float16 tensor on CPU, or None
      lens_gaps      : list of z_B - z_A per layer via norm+lm_head, or None
    """
    inp = tok(forced_prompt, return_tensors="pt").to(device)
    out = model(**inp, output_hidden_states=capture_residuals or capture_lens)

    logits_last = out.logits[0, -1, :].float()
    zA = _group_logit(logits_last, ids_A)
    zB = _group_logit(logits_last, ids_B)

    residuals, lens_gaps = None, None
    if capture_residuals or capture_lens:
        # hidden_states: tuple of (1, seq, d) with n_layers+1 entries (h_0 = embeddings)
        hs = [h[0, -1, :] for h in out.hidden_states]
        if capture_residuals:
            residuals = torch.stack(hs).to(torch.float16).cpu()
        if capture_lens:
            norm = _find_norm(model)
            head = model.get_output_embeddings()
            lens_gaps = []
            if norm is not None and head is not None:
                for h in hs:
                    v = norm(h.unsqueeze(0)).squeeze(0)
                    z = head(v).float()
                    lens_gaps.append(_group_logit(z, ids_B) - _group_logit(z, ids_A))
            else:
                lens_gaps = None

    return {
        "logit_A": zA,
        "logit_B": zB,
        "gap_raw": zB - zA,
        "residuals": residuals,
        "lens_gaps": lens_gaps,
    }


ANSWER_RE = re.compile(r"Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}")


def parse_answer_letter(text):
    """Extract the model's real final A/B answer from generated CoT text.

    Takes the LAST regex match, not the first: the cot template's own format
    instructions are frequently echoed verbatim by the model before its real
    answer, and a first-match search silently locks onto that instruction
    text instead. Shared by the live generation path (post_cot_generate) and
    offline re-parsing of already-collected cot_text (analysis/reparse_answers.py) --
    both must agree, or aggregation will disagree with fresh generation.
    """
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1) or m.group(2)


@torch.no_grad()
def post_cot_generate(model, tok, device, cot_prompt, ids_A, ids_B, max_new_tokens=2048,
                       sampling=None, seed=None):
    """Generate the CoT response, then locate the actual A/B answer token and read
    the logit gap AT that generation step.

    valid=False when the model never writes 'Answer: A/B'. Invalid rounds carry NO
    logit gap -- the scores[-1] fallback of earlier pipelines measured the logit
    after an arbitrary final prose token and is scientifically meaningless.

    sampling: optional dict e.g. {"do_sample": True, "temperature": 0.6, "top_p": 0.95}.
    R1-distill models need this per DeepSeek's own usage recommendations -- greedy
    decoding is the documented cause of the endless-repetition failure mode. seed
    pins the global torch RNG so a sampled run stays reproducible under
    --skip-existing (this transformers version's generate() rejects a `generator`
    kwarg -- it's not in the validated model_kwargs allowlist -- so we seed globally
    right before the call instead).
    """
    inp = tok(cot_prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
    )
    if sampling:
        gen_kwargs.update(sampling)
        if seed is not None:
            torch.manual_seed(seed)
    out = model.generate(**inp, **gen_kwargs)
    gen_ids = out.sequences[0][inp["input_ids"].shape[1]:].tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)

    # Use out.logits (raw, pre-logits-processor) everywhere below, NOT
    # out.scores. Under sampling, scores are POST top_p/temperature filtering
    # -- tokens outside the retained mass are set to -inf, so a token that
    # simply wasn't sampled at a given step can silently read as -inf even
    # when the model assigned it real probability mass. This is fine for
    # greedy models (do_sample=False applies no filtering, scores==logits)
    # but for dsqwen7/dsllama70 it previously corrupted BOTH this diagnostic
    # (NaN tail gaps -- both A and B filtered out mid-reasoning, since
    # single-letter tokens are rarely in the top-p mass mid-sentence) and,
    # more seriously, gap_raw itself: at the actual answer step, the letter
    # that WAS generated must have been retained, but the other letter could
    # still be filtered to -inf, making a "valid" round's logit gap an
    # artifact of the top-p cutoff rather than the model's real preference.

    # Diagnostic ONLY -- mean/std of the A/B logit gap over the last TAIL_WINDOW
    # generation steps, computed for every round (valid or not). This is NOT a
    # decision proxy and must never feed delta_post_sem/decision_act1 (that's
    # exactly the scores[-1]-fallback mistake this module's docstring warns
    # against). Purpose: test whether a round that never emits an answer marker
    # (post_valid=False, e.g. a stuck repetition loop) reflects genuine internal
    # ambiguity (tail gap ~0) or a firm-but-unformatted commitment (tail gap
    # consistently one-sided) -- i.e. is the behavioral non-answer mirrored by
    # representational ambiguity, or is it a pure formatting failure.
    TAIL_WINDOW = 50
    tail_logits = out.logits[-min(TAIL_WINDOW, len(out.logits)):]
    tail_gaps = [_group_logit(s[0].float(), ids_B) - _group_logit(s[0].float(), ids_A)
                 for s in tail_logits]
    tail_gap_mean = sum(tail_gaps) / len(tail_gaps) if tail_gaps else None
    tail_gap_std = ((sum((g - tail_gap_mean) ** 2 for g in tail_gaps) / len(tail_gaps)) ** 0.5
                     if tail_gaps else None)

    letter = parse_answer_letter(text)
    result = {
        "text": text,
        "n_gen_tokens": len(gen_ids),
        "valid": False,
        "parsed_letter": None,
        "gap_raw": None,
        "answer_step": None,
        "tail_gap_mean": tail_gap_mean,
        "tail_gap_std": tail_gap_std,
    }
    if letter is None:
        return result

    target_ids = ids_A if letter == "A" else ids_B
    # scan generated ids for the letter token; take the LAST occurrence that decodes
    # to the answer letter (final answers are written at the end of the CoT)
    answer_step = None
    for step, tid in enumerate(gen_ids):
        if tid in target_ids:
            answer_step = step
    if answer_step is None:
        return result

    step_logits = out.logits[answer_step][0].float()
    zA = _group_logit(step_logits, ids_A)
    zB = _group_logit(step_logits, ids_B)
    result.update({
        "valid": True,
        "parsed_letter": letter,
        "gap_raw": zB - zA,
        "answer_step": answer_step,
        "logit_A": zA,
        "logit_B": zB,
    })
    return result
