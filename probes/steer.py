"""P4: causal steering / rescue experiment.

The P1 probes show the hidden state is linearly decodable from the residual
stream at the pre-CoT measurement point, while the pre-CoT preference
(delta_pre_sem) often disagrees with the oracle. This runner tests whether the
decoded direction is CAUSALLY upstream of the preference: it replays the exact
sessions of the main sweep (same env seeds -> same scripted histories, same
forced prompts) and re-runs the pre-CoT probe while adding a steering vector to
the residual stream at the probe's best layer, at the last prompt position.

Conditions per round (one batched forward per label mapping):
  baseline          no intervention (sanity: must reproduce delta_pre_sem)
  rescue_a{alpha}   +alpha * sigma_proj * w_hat, signed TOWARD the oracle action
  anti_a{alpha}     the same vector signed AWAY from the oracle action
  rand{j}_a{alpha}  random unit directions at the same magnitudes (specificity)

sigma_proj is the pooled std of residual projections onto w_hat (saved by
probe_train.py --weights-only), so alpha is in units of natural variation along
the probe direction.

Headline stats (analysis/steer_summary.py):
  rescue rate  fraction of unambiguous wrong-signed rounds whose semantic gap
               flips to oracle-correct under rescue steering
  break rate   fraction of correct rounds flipped by anti steering
  random rate  flips under random directions (should stay near zero)

Usage:
  python run_steer.py --models llama --envs ipd --seeds 0 1 2
  python run_steer.py --models qwen27 --envs monty bandit stag_hunt ipd trust_game \
                      --seeds $(seq 0 39) --tag P1_v3 --skip-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from core.model_loader import MODELS, load_model, unload_model, flush, apply_chat, answer_token_ids
from run_all import load_template, render, semantic_gap
from envs import make_env, ENVS


# ---------------------------------------------------------------- directions

def load_direction(tag, model_key, env_name):
    path = HERE / "results" / tag / f"direction_{model_key}_{env_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run: python analysis/probe_train.py --tag {tag} "
            f"--residuals-dir residuals --models {model_key} --envs {env_name} --weights-only")
    d = np.load(path)
    return {
        "w_unit": d["w_unit"].astype(np.float32),
        "layer": int(d["layer"]),
        "sigma_proj": float(d["sigma_proj"]),
    }


def random_unit_dirs(n, dim, model_key, env_name):
    """Fixed per-combo seed so reruns and resumed jobs use identical controls."""
    seed = abs(hash((model_key, env_name, "steer-controls"))) % (2**32)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, dim)).astype(np.float32)
    return R / np.linalg.norm(R, axis=1, keepdims=True)


# ---------------------------------------------------------------- hook plumbing

def _find_decoder_stack(model):
    """Locate the text-decoder ModuleList and its owner across architectures,
    including multimodal wrappers (e.g. gemma-4's language_model nesting) where
    the stack is not at model.model.layers. Picks the longest ModuleList of
    *DecoderLayer modules, which skips vision encoders."""
    import torch.nn as nn
    best = None
    for _, mod in model.named_modules():
        ml = getattr(mod, "layers", None)
        if (isinstance(ml, nn.ModuleList) and len(ml)
                and "DecoderLayer" in type(ml[0]).__name__):
            if best is None or len(ml) > len(best[1]):
                best = (mod, ml)
    if best is None:
        raise RuntimeError("cannot locate decoder layer list on this architecture")
    return best


def resolve_hook_module(model, hs_index):
    """Module whose OUTPUT is hidden_states[hs_index] of the probe capture:
    index 0 = embeddings, index k (1..L-1) = output of decoder block k-1,
    index L = output of the final norm."""
    base, layers = _find_decoder_stack(model)
    L = len(layers)
    if hs_index == 0:
        print("  [warn] steering at hs index 0 (embeddings); pre-scaling on some "
              "architectures (gemma) makes this only approximate", flush=True)
        return model.get_input_embeddings()
    if hs_index <= L - 1:
        return layers[hs_index - 1]
    norm = (getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
            or getattr(base, "ln_f", None))
    if norm is None:
        raise RuntimeError(f"hs index {hs_index} maps to the final norm, not found")
    return norm


def make_hook(steer_bd, positions):
    """steer_bd: (batch, d) float32 CPU tensor, row i added to sample i."""
    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        s = steer_bd.to(device=out.device, dtype=out.dtype)
        if positions == "all":
            out += s[:, None, :]
        else:
            out[:, -1, :] += s
        return output
    return hook


# ---------------------------------------------------------------- probe pass

def _group_logits_batch(logits_bv, ids):
    return torch.logsumexp(logits_bv[:, ids], dim=1)


@torch.no_grad()
def steered_gaps(model, tok, device, prompt, ids_A, ids_B, hook_module,
                 steer_mat, positions, max_batch):
    """One forward per chunk of conditions; rows of steer_mat share the prompt.
    Returns gap_raw (z_B - z_A) per condition row."""
    inp = tok(prompt, return_tensors="pt").to(device)
    gaps = []
    for lo in range(0, len(steer_mat), max_batch):
        chunk = steer_mat[lo:lo + max_batch]
        ids = inp["input_ids"].expand(len(chunk), -1)
        mask = inp["attention_mask"].expand(len(chunk), -1)
        handle = hook_module.register_forward_hook(
            make_hook(torch.from_numpy(chunk), positions))
        try:
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            handle.remove()
        logits_last = out.logits[:, -1, :].float()
        zA = _group_logits_batch(logits_last, ids_A)
        zB = _group_logits_batch(logits_last, ids_B)
        gaps.extend((zB - zA).tolist())
        del out
    return gaps


# ---------------------------------------------------------------- session replay

def steer_session(model, tok, device, model_key, env_name, seed, T, direction,
                  alphas, n_random, positions, max_batch, out_dir):
    rng = np.random.default_rng(seed)
    env = make_env(env_name, rng, T, seed=seed)
    forced_tmpl = load_template(env_name, "forced")
    ids_A, ids_B = answer_token_ids(tok)

    w = direction["w_unit"]
    sigma = direction["sigma_proj"]
    layer = direction["layer"]
    hook_module = resolve_hook_module(model, layer)
    R = random_unit_dirs(n_random, len(w), model_key, env_name)

    # condition names; the vector for each is built per round (oracle sign)
    names = ["baseline"]
    for a in alphas:
        names += [f"rescue_a{a:g}", f"anti_a{a:g}"]
    for j in range(n_random):
        for a in alphas:
            names.append(f"rand{j}_a{a:g}")

    # stored pre-sweep gaps, for the baseline sanity check
    stored = {}
    resp_path = HERE / "responses" / f"{model_key}_{env_name}_seed{seed}.json"
    if resp_path.exists():
        for r in json.loads(resp_path.read_text())["rounds"]:
            stored[r["t"]] = r["delta_pre_sem"]

    records, rounds_out = [], []
    print(f"\n=== steer {model_key} | {env_name} | seed={seed} | layer={layer} "
          f"| {env.hidden_param()} ===", flush=True)

    for t in range(1, T + 1):
        ctx = env.new_round(t)
        history = env.format_history(records)
        oracle_p = env.oracle_p_act1(records)
        gt_act1 = oracle_p > 0.5
        sign = 1.0 if gt_act1 else -1.0

        rows = [np.zeros_like(w)]
        for a in alphas:
            rows += [sign * a * sigma * w, -sign * a * sigma * w]
        for j in range(n_random):
            for a in alphas:
                rows.append(sign * a * sigma * R[j])
        steer_mat = np.stack(rows)

        gaps_sem = np.zeros(len(names))
        for mapping in (0, 1):
            user_text = render(forced_tmpl, env, t, T, history, ctx, mapping)
            prompt = apply_chat(tok, user_text, assistant_prefill="Answer:")
            raw = steered_gaps(model, tok, device, prompt, ids_A, ids_B,
                               hook_module, steer_mat, positions, max_batch)
            gaps_sem += np.array([semantic_gap(g, mapping) for g in raw])
        gaps_sem /= 2.0

        by_cond = dict(zip(names, [float(g) for g in gaps_sem]))
        rounds_out.append({
            "t": t,
            "oracle_p_act1": oracle_p,
            "gt_act1": bool(gt_act1),
            "unambiguous": abs(oracle_p - 0.5) > 0.1,
            "stored_delta_pre_sem": stored.get(t),
            "gaps": by_cond,
        })

        beh = env.behavior_action(t)
        _, rec = env.outcome(t, ctx, beh)
        records.append(rec)
        flush(device)

    session = {
        "model": MODELS[model_key]["id"],
        "model_key": model_key,
        "env": env_name,
        "seed": seed,
        "T": T,
        "hidden_param": env.hidden_param(),
        "steer_layer": layer,
        "sigma_proj": sigma,
        "alphas": list(alphas),
        "n_random": n_random,
        "positions": positions,
        "conditions": names,
        "rounds": rounds_out,
    }
    out_path = out_dir / f"{model_key}_{env_name}_seed{seed}.json"
    out_path.write_text(json.dumps(session, indent=2))

    # quick in-log tally at the largest alpha
    a_max = max(alphas)
    wrong = [r for r in rounds_out
             if r["unambiguous"] and (r["gaps"]["baseline"] > 0) != r["gt_act1"]]
    rescued = [r for r in wrong
               if (r["gaps"][f"rescue_a{a_max:g}"] > 0) == r["gt_act1"]]
    drift = [abs(r["gaps"]["baseline"] - r["stored_delta_pre_sem"])
             for r in rounds_out if r["stored_delta_pre_sem"] is not None]
    print(f"  -> wrong={len(wrong)}  rescued@a{a_max:g}={len(rescued)}  "
          f"baseline-replay max|drift|={max(drift):.3f}" if drift else
          f"  -> wrong={len(wrong)}  rescued@a{a_max:g}={len(rescued)}", flush=True)
    return session


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="P4 steering / rescue experiment")
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(40)))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--tag", default="P1_v3",
                    help="directions are read from results/{tag}/, outputs go to steer/{tag}/")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--positions", choices=["last", "all"], default="last")
    ap.add_argument("--max-batch", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out_dir = HERE / "steer" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for model_key in args.models:
        directions = {}
        for env_name in args.envs:
            directions[env_name] = load_direction(args.tag, model_key, env_name)

        todo = []
        for env_name in args.envs:
            for seed in args.seeds:
                out_path = out_dir / f"{model_key}_{env_name}_seed{seed}.json"
                if args.skip_existing and out_path.exists():
                    try:
                        json.loads(out_path.read_text())
                        continue
                    except json.JSONDecodeError:
                        pass
                todo.append((env_name, seed))
        if not todo:
            print(f"[resume] {model_key}: all sessions already done")
            continue
        print(f"{model_key}: {len(todo)} sessions to steer")

        model, tok, device = load_model(MODELS[model_key]["id"], MODELS[model_key])
        try:
            for env_name, seed in todo:
                steer_session(model, tok, device, model_key, env_name, seed,
                              args.rounds, directions[env_name], args.alphas,
                              args.n_random, args.positions, args.max_batch, out_dir)
        finally:
            unload_model(model, tok, device)
            del model, tok
            flush(device)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
