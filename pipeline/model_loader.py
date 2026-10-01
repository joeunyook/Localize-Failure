"""Model loading / unloading, shared with the Monty pipeline conventions."""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "llama":   {"id": "meta-llama/Llama-3.2-3B-Instruct", "label": "Llama-3.2-3B-Instruct", "color": "#1F77B4"},
    "phi":     {"id": "microsoft/Phi-4-mini-instruct",    "label": "Phi-4-mini-instruct",   "color": "#FF7F0E"},
    "qwen":    {"id": "Qwen/Qwen3.5-4B",                  "label": "Qwen3.5-4B",            "color": "#9467BD"},
    # midscale additions (keys must stay underscore-free: probe filenames parse on "_")
    "qwen27":  {"id": "Qwen/Qwen3.6-27B",                 "label": "Qwen3.6-27B",           "color": "#2CA02C"},
    "gemma31": {"id": "google/gemma-4-31B-it",            "label": "gemma-4-31B-it",        "color": "#D62728"},
    # 2026-07-13 additions.
    # gemma2b/dsqwen7: cheap scale anchors (cross-family, so scale comparisons
    # against them are confounded -- the controlled pair is llama70/dsllama70,
    # same base weights +- R1 reasoning distillation).
    # prefill_prefix: R1-distill templates open a <think> block; the forced
    # probe must close it so 'Answer:' is the model's next real token (the
    # pre-CoT measurement point). Applied only when a prefill is given -- the
    # post-CoT generation run keeps thinking enabled on purpose.
    # multi_gpu: ~140GB bf16 weights, needs device_map=auto across >=2 GH200s.
    "gemma2b":   {"id": "google/gemma-2-2b-it",                            "label": "gemma-2-2b-it",          "color": "#E377C2"},
    # R1-Distill series: DeepSeek's own model card mandates non-greedy decoding
    # ("temperature within 0.5-0.7, 0.6 recommended, to prevent endless
    # repetitions or incoherent outputs") -- greedy (do_sample=False) is exactly
    # the failure mode that warning describes, and it's what produced the
    # dsqwen7 repetition-loop failures seen in P3. Applies to post-CoT
    # generation only; the pre-CoT forced single-token read is unaffected by
    # sampling since it never calls .generate().
    "dsqwen7":   {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",         "label": "R1-Distill-Qwen-7B",     "color": "#8C564B",
                  "prefill_prefix": "<think>\n\n</think>\n\n",
                  "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.95},
                  "answer_format": "boxed"},
    # Load from the Hugging Face hub, or replace with a local path.
    "llama70":   {"id": "meta-llama/Llama-3.3-70B-Instruct", "label": "Llama-3.3-70B-Instruct", "color": "#17BECF",
                  "multi_gpu": True},
    # tokenizer_patch_source: DeepSeek's published tokenizer.json for this repo
    # ships without a pre_tokenizer (confirmed via fresh re-download, same md5
    # as the local cache -- this is upstream, not a corrupted download). Encode
    # silently drops whitespace ('hello' and ' hello' tokenize identically),
    # which made every post-CoT generation undecodable (format_rate=0%).
    # R1-Distill-Llama-70B is a fine-tune of Llama-3.3-70B-Instruct with an
    # unchanged vocab, so we borrow that sibling's pre_tokenizer/decoder at
    # load time and keep DeepSeek's own tokenizer for everything else
    # (chat template, <think> special tokens) -- verified round-trip decode
    # and chat template both correct after the patch.
    "dsllama70": {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",       "label": "R1-Distill-Llama-70B",   "color": "#BCBD22",
                  "multi_gpu": True, "prefill_prefix": "<think>\n\n</think>\n\n",
                  "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.95},
                  "answer_format": "boxed",
                  "tokenizer_patch_source": "meta-llama/Llama-3.3-70B-Instruct"},
}


def load_model(model_id, cfg=None):
    cfg = cfg or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if (cfg.get("multi_gpu") and device == "cuda") else device
    print(f"\nLoading {model_id}  ->  device_map={device_map}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    patch_src = cfg.get("tokenizer_patch_source")
    if patch_src:
        print(f"  patching broken pre_tokenizer/decoder from {patch_src}", flush=True)
        base_tok = AutoTokenizer.from_pretrained(patch_src)
        tok.backend_tokenizer.pre_tokenizer = base_tok.backend_tokenizer.pre_tokenizer
        tok.backend_tokenizer.decoder = base_tok.backend_tokenizer.decoder
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", attn_implementation="eager", device_map=device_map,
    )
    model.eval()
    # carried on the tokenizer so apply_chat/generate call sites don't need the registry
    tok.prefill_prefix = cfg.get("prefill_prefix")
    tok.sampling = cfg.get("sampling")
    tok.answer_format = cfg.get("answer_format")  # None (letter) or "boxed"
    return model, tok, device


def unload_model(model, tok, device):
    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def flush(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def apply_chat(tok, user_text, assistant_prefill=None):
    """Render a single-turn chat prompt. enable_thinking=False is passed when the
    tokenizer supports it (Qwen hybrid-reasoning models); others fall back silently.

    assistant_prefill, when given, is appended AFTER the generation prompt so the
    model's next token continues the assistant turn from that exact string. This is
    how the forced-choice probe pins the measurement point at 'Answer:'.
    """
    messages = [{"role": "user", "content": user_text}]
    try:
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if assistant_prefill:
        prefix = getattr(tok, "prefill_prefix", None)
        if prefix:
            if formatted.rstrip().endswith("<think>"):
                # template already opened the think block: just close it
                formatted = formatted + "\n\n</think>\n\n"
            else:
                formatted = formatted + prefix
        formatted = formatted + assistant_prefill
    return formatted


def answer_token_ids(tok):
    """Token-id variant sets for the answer letters.

    After 'Answer:' the next token is usually ' A' (with leading space), but some
    tokenizers emit bare 'A'. We collect all single-token encodings of both variants
    and aggregate with logsumexp at probe time.
    """
    def variants(letter):
        ids = set()
        for s in (f" {letter}", letter):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    ids_A, ids_B = variants("A"), variants("B")
    assert ids_A and ids_B, "tokenizer produced no single-token A/B variants"
    return ids_A, ids_B
