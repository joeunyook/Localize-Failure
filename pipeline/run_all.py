"""Master runner for the logit-probe pipeline.

Per (model, env, seed) session, for every round t = 1..T:

  1. Build the forced-choice prompt under BOTH label mappings (A/B swapped) and
     run the pre-CoT probe on each. The semantic pre-CoT gap
         delta_pre_sem = mean over mappings of [z(act1 letter) - z(act0 letter)]
     cancels any intrinsic letter bias (models often prefer the token 'A').
  2. Run the post-CoT generation probe on the canonical mapping (A=act0, B=act1)
     and read the answer-token gap. Rounds with no generated 'Answer: A/B' are
     valid=False and excluded from logit metrics -- never fall back to scores[-1].
  3. Record the environment's exact Bayesian oracle P(act1 optimal | h_t).
  4. Save the residual stream h_k[last_prompt_token] (all layers, canonical
     mapping only) for offline linear-probe training.
  5. Advance the history with the SCRIPTED behavior policy (the model observes,
     it does not play into the evidence stream).

Usage:
  python run_all.py --models llama phi qwen --envs monty bandit stag_hunt ipd trust_game \
                    --seeds 0 1 2 --rounds 10
  python run_all.py --models llama --envs monty --seeds 0 --rounds 10 --no-post   # probes only, fast
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from core.model_loader import MODELS, load_model, unload_model, flush, apply_chat, answer_token_ids
from core.probes import pre_cot_probe, post_cot_generate
from core.metrics import summarize_session, sigmoid
from core.io_utils import save_json, save_residuals
from envs import make_env, ENVS

MAX_ANSWER_TOKENS = 2048


def load_template(env_name, kind):
    return (HERE / "prompts" / f"{env_name}_{kind}.txt").read_text()


# Default answer-format instruction. R1-distill models get DeepSeek's own
# recommended phrasing instead (see MODELS["dsqwen7"/"dsllama70"]["answer_format"]
# in core/model_loader.py): "Please reason step by step, and put your final
# answer within \boxed{}." -- their card gives this for math problems, but it's
# a format-compliance instruction, not a math-specific one, and matches what
# these models do natively anyway (llama70, a non-R1 model, spontaneously wrote
# \boxed{} unprompted). core.probes.ANSWER_RE accepts both forms from any model.
ANSWER_FORMAT_LETTER = "Write your final answer — exactly one of:\n     Answer: A\n     Answer: B"
ANSWER_FORMAT_BOXED = "Please reason step by step, and put your final answer within \\boxed{}, exactly one of:\n     \\boxed{A}\n     \\boxed{B}"


def render(template, env, t, T, history, ctx, mapping, answer_format=ANSWER_FORMAT_LETTER):
    """mapping 0: A=act0, B=act1 (canonical). mapping 1: swapped."""
    if mapping == 0:
        a_desc, b_desc = env.act0_desc, env.act1_desc
    else:
        a_desc, b_desc = env.act1_desc, env.act0_desc
    fields = {
        "t": t, "T": T, "history": history,
        "option_A_desc": a_desc, "option_B_desc": b_desc,
        "answer_format": answer_format,
        **ctx,
    }
    return template.format(**fields)


def semantic_gap(raw_gap, mapping):
    """raw gap is z_B - z_A in label space; flip sign when act1 sits on A."""
    return raw_gap if mapping == 0 else -raw_gap


def round_seed(model_key, env_name, seed, t):
    """Deterministic torch generator seed for sampled post-CoT generation, so a
    resumed/rerun (--skip-existing) round is reproducible despite do_sample=True."""
    key = f"{model_key}|{env_name}|{seed}|{t}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def run_session(model, tok, device, model_key, env_name, seed, T,
                do_post=True, save_resid=True, out_root=HERE, forced_variant=None, no_tally=False):
    rng = np.random.default_rng(seed)
    env = make_env(env_name, rng, T, seed=seed)
    cot_tmpl = load_template(env_name, "cot" if forced_variant is None
                             else f"cot_{forced_variant}")
    forced_tmpl = load_template(env_name, "forced" if forced_variant is None
                                else f"forced_{forced_variant}")
    ids_A, ids_B = answer_token_ids(tok)

    records = []       # environment history (scripted)
    rounds_out = []    # per-round measurements
    residual_stack = []

    print(f"\n=== {model_key} | {env_name} | seed={seed} | {env.hidden_param()} ===", flush=True)

    for t in range(1, T + 1):
        ctx = env.new_round(t)
        history = env.format_history(records)
        if no_tally:
            history = "\n".join(l for l in history.splitlines() if not l.startswith("# Tally"))
        oracle_p = env.oracle_p_act1(records)

        # ---- pre-CoT forced-choice probe, both label mappings ----
        pre_gaps_sem = []
        resid = None
        for mapping in (0, 1):
            user_text = render(forced_tmpl, env, t, T, history, ctx, mapping)
            prompt = apply_chat(tok, user_text, assistant_prefill="Answer:")
            res = pre_cot_probe(
                model, tok, device, prompt, ids_A, ids_B,
                capture_residuals=(save_resid and mapping == 0),
                capture_lens=(mapping == 0),
            )
            pre_gaps_sem.append(semantic_gap(res["gap_raw"], mapping))
            if mapping == 0:
                resid = res["residuals"]
                lens_gaps = res["lens_gaps"]
        delta_pre_sem = float(np.mean(pre_gaps_sem))
        letter_bias = float((pre_gaps_sem[0] - pre_gaps_sem[1]) / 2.0)
        if resid is not None:
            residual_stack.append(resid.numpy())

        # ---- post-CoT generation probe, canonical mapping ----
        post = {"valid": False, "gap_raw": None, "parsed_letter": None,
                "text": None, "n_gen_tokens": 0,
                "tail_gap_mean": None, "tail_gap_std": None}
        if do_post:
            fmt = ANSWER_FORMAT_BOXED if getattr(tok, "answer_format", None) == "boxed" else ANSWER_FORMAT_LETTER
            user_text = render(cot_tmpl, env, t, T, history, ctx, 0, answer_format=fmt)
            prompt = apply_chat(tok, user_text)
            post = post_cot_generate(model, tok, device, prompt, ids_A, ids_B,
                                     max_new_tokens=MAX_ANSWER_TOKENS,
                                     sampling=getattr(tok, "sampling", None),
                                     seed=round_seed(model_key, env_name, seed, t))
        delta_post_sem = post["gap_raw"] if post["valid"] else None  # mapping 0: B=act1

        # behavioral decision from the parsed CoT answer (canonical mapping)
        if post["valid"]:
            decision_act1 = post["parsed_letter"] == "B"
        else:
            decision_act1 = None

        rounds_out.append({
            "t": t,
            "oracle_p_act1": oracle_p,
            "delta_pre_sem": delta_pre_sem,
            "delta_pre_by_mapping": pre_gaps_sem,
            "letter_bias": letter_bias,
            "delta_post_sem": delta_post_sem,
            "delta_amp": (delta_post_sem - delta_pre_sem) if delta_post_sem is not None else None,
            "post_valid": bool(post["valid"]),
            "decision_act1": decision_act1,
            "parsed_letter": post["parsed_letter"],
            "n_gen_tokens": post["n_gen_tokens"],
            "tail_gap_mean": post["tail_gap_mean"],
            "tail_gap_std": post["tail_gap_std"],
            "lens_gaps": lens_gaps,
            "round_ctx": {k: v for k, v in ctx.items() if k != "prize"},
            "cot_text": post["text"],
        })

        pi_pre = sigmoid(delta_pre_sem)
        pv = f"{delta_post_sem:+.2f}" if delta_post_sem is not None else "INVALID"
        print(f"  t={t:2d}  oracle={oracle_p:.2f}  pre={delta_pre_sem:+.2f} "
              f"(pi={pi_pre:.2f})  post={pv}  gen_toks={post['n_gen_tokens']}", flush=True)

        # ---- advance scripted history ----
        beh = env.behavior_action(t)
        _, rec = env.outcome(t, ctx, beh)
        records.append(rec)
        flush(device)

    summary = summarize_session(rounds_out)
    session = {
        "model": MODELS[model_key]["id"],
        "model_key": model_key,
        "env": env_name,
        "seed": seed,
        "T": T,
        "hidden_param": env.hidden_param(),
        "act0": env.act0,
        "act1": env.act1,
        "summary": summary,
        "history_records": records,
    }

    stem = f"{model_key}_{env_name}_seed{seed}"
    save_json(session, out_root / "results" / f"{stem}.json")
    save_json({"session": {k: v for k, v in session.items() if k != "history_records"},
               "rounds": rounds_out}, out_root / "responses" / f"{stem}.json")
    if residual_stack:
        (out_root / "residuals").mkdir(parents=True, exist_ok=True)
        np.save(out_root / "residuals" / f"{stem}.npy",
                np.stack(residual_stack))  # (T, n_layers+1, d_model) float16
        labels = [{"t": r["t"], "oracle_p_act1": r["oracle_p_act1"],
                   "gt_act1": r["oracle_p_act1"] > 0.5} for r in rounds_out]
        save_json({"hidden_param": env.hidden_param(), "labels": labels},
                  out_root / "residuals" / f"{stem}_labels.json")

    print(f"  -> Delta_T={summary['Delta_T']}  format={summary['format_rate']:.0%}  "
          f"drift_corr_pre={summary['drift_corr_pre']}", flush=True)
    return session


def session_done(model_key, env_name, seed, save_resid, out_root=HERE):
    """True iff every artifact of this session is on disk and the file written
    last (residual labels, or responses when residuals are off) parses -- so a
    job killed mid-session is treated as not-done and the session reruns."""
    import json
    stem = f"{model_key}_{env_name}_seed{seed}"
    paths = [out_root / "results" / f"{stem}.json", out_root / "responses" / f"{stem}.json"]
    if save_resid:
        paths += [out_root / "residuals" / f"{stem}.npy",
                  out_root / "residuals" / f"{stem}_labels.json"]
    if not all(p.exists() for p in paths):
        return False
    try:
        json.loads(paths[-1].read_text()) if paths[-1].suffix == ".json" else None
        return True
    except (json.JSONDecodeError, OSError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "phi", "qwen"],
                    choices=list(MODELS))
    ap.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--no-post", action="store_true",
                    help="skip CoT generation (pre-CoT probes only; ~50x faster)")
    ap.add_argument("--no-residuals", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip sessions whose outputs are already complete on disk "
                         "(resume after a cut job)")
    ap.add_argument("--out-root", type=Path, default=HERE,
                    help="root directory for results/, responses/, residuals/ -- "
                         "use a separate root (e.g. P3_postcot) so a post-CoT "
                         "campaign never collides with the canonical pre-CoT data")
    ap.add_argument("--forced-variant", default=None,
                    help="load prompts/{env}_forced_{variant}.txt instead of the "
                         "canonical {env}_forced.txt -- for prompt-framing ablations "
                         "(e.g. --forced-variant dry). cot template is unaffected.")
    ap.add_argument("--no-tally", action="store_true",
                    help="strip # Tally line from history -- tally-ablation control")
    args = ap.parse_args()

    t0 = time.time()
    for model_key in args.models:
        save_resid = not args.no_residuals
        todo = [(e, s) for e in args.envs for s in args.seeds
                if not (args.skip_existing and
                        session_done(model_key, e, s, save_resid, args.out_root))]
        n_skipped = len(args.envs) * len(args.seeds) - len(todo)
        if n_skipped:
            print(f"[resume] {model_key}: skipping {n_skipped} completed sessions, "
                  f"{len(todo)} to go", flush=True)
        if not todo:
            continue
        model, tok, device = load_model(MODELS[model_key]["id"], MODELS[model_key])
        try:
            for env_name, seed in todo:
                run_session(model, tok, device, model_key, env_name, seed,
                            args.rounds, do_post=not args.no_post,
                            save_resid=save_resid, out_root=args.out_root,
                            forced_variant=args.forced_variant,
                            no_tally=args.no_tally)
        finally:
            unload_model(model, tok, device)
            # unload_model only drops its own references; without dropping ours
            # the weights stay on-GPU and the next model's load OOMs
            del model, tok
            flush(device)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
