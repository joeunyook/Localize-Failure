"""P1: offline linear probes on the saved residual streams.

For each (model, env), pool residuals across seeds and rounds, and train one
logistic-regression probe PER LAYER to predict the normative optimal action
(gt_act1 = oracle_p > 0.5). Reports leave-one-seed-out cross-validated accuracy
per layer -- the 'representational separability' curve.

Interpretation:
  - If accuracy rises with layer depth and beats the majority baseline well
    before the final layer, the model ENCODES the correct belief internally.
  - If accuracy stays at baseline across all layers, the accumulated history
    never enters the representation (belief-update failure).
  - Compare against P2 (drift_corr_pre) to dissociate knowing from acting:
    high probe accuracy + wrong-signed P2 = belief present, mapping broken.

Usage:
  python analysis/probe_train.py                          # all combos found
  python analysis/probe_train.py --models llama --envs monty
  python analysis/probe_train.py --residuals-dir residuals/P2_v1 --tag P1_v1

Outputs (under --tag, default P1_v1):
  results/{tag}/probe_{model}_{env}.json   per-layer LOSO accuracy + baseline
  results/{tag}/probe_summary.json         one row per combo (best layer etc.)
  plots/{tag}/probe_{model}_{env}.png      accuracy-vs-layer curve
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sys.exit("pip install scikit-learn")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

AMBIG_EPS = 0.10   # drop rounds with |oracle - 0.5| <= this (labels are noise)
PROBE_C = 0.1      # strong L2: d_model >> n_samples


def load_pool(residuals_dir, model_key, env_name):
    """Return {seed: X (rounds, L+1, d)}, {seed: y (rounds,)} with ambiguous rounds dropped."""
    X_by_seed, y_by_seed = {}, {}
    for npy in sorted(residuals_dir.glob(f"{model_key}_{env_name}_seed*.npy")):
        seed = int(npy.stem.split("seed")[-1])
        labels_path = npy.with_name(npy.stem + "_labels.json")
        if not labels_path.exists():
            continue
        labels = json.loads(labels_path.read_text())["labels"]
        X = np.load(npy).astype(np.float32)
        y = np.array([l["gt_act1"] for l in labels])
        keep = np.array([abs(l["oracle_p_act1"] - 0.5) > AMBIG_EPS for l in labels])
        X_by_seed[seed], y_by_seed[seed] = X[keep], y[keep]
    return X_by_seed, y_by_seed


def loso_accuracy(X_by_seed, y_by_seed, layer):
    """Leave-one-seed-out CV for one layer; returns (acc, acc_bal).

    acc      mean over folds of held-out accuracy (definition unchanged from
             the P1_v3 runs, so regenerated numbers stay comparable).
    acc_bal  macro-average over the two label classes of the POOLED held-out
             predictions: 0.5 for any constant predictor regardless of label
             skew -- the artifact-proof twin of the behavior tables'
             balanced agreement. None if a class is absent from the pool.
    (None, None) if < 2 seeds usable.
    """
    seeds = sorted(X_by_seed)
    if len(seeds) < 2:
        return None, None
    accs, y_all, correct_all = [], [], []
    for held in seeds:
        Xtr = np.concatenate([X_by_seed[s][:, layer, :] for s in seeds if s != held])
        ytr = np.concatenate([y_by_seed[s] for s in seeds if s != held])
        Xte, yte = X_by_seed[held][:, layer, :], y_by_seed[held]
        if len(set(ytr)) < 2 or len(Xte) == 0:
            continue
        clf = LogisticRegression(max_iter=2000, C=PROBE_C)
        clf.fit(Xtr, ytr)
        correct = clf.predict(Xte) == yte
        accs.append(float(correct.mean()))
        y_all.append(np.asarray(yte, bool))
        correct_all.append(correct)
    if not accs:
        return None, None
    y = np.concatenate(y_all)
    c = np.concatenate(correct_all)
    bal = None
    if y.any() and (~y).any():
        bal = float((c[y].mean() + c[~y].mean()) / 2.0)
    return float(np.mean(accs)), bal


def majority_baseline(y_by_seed):
    y = np.concatenate(list(y_by_seed.values()))
    return float(max(y.mean(), 1 - y.mean()))


def fit_direction(X_by_seed, y_by_seed, layer):
    """Refit one probe on ALL pooled data at `layer` and return the steering
    direction. Oriented so positive projection predicts act1 (gt_act1=True).
    sigma_proj is the pooled std of projections onto the unit direction --
    steering magnitudes are expressed in multiples of it."""
    X = np.concatenate([X_by_seed[s][:, layer, :] for s in sorted(X_by_seed)])
    y = np.concatenate([y_by_seed[s] for s in sorted(y_by_seed)])
    clf = LogisticRegression(max_iter=2000, C=PROBE_C)
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float32)
    b = float(clf.intercept_[0])
    if not bool(clf.classes_[1]):        # ensure w points toward act1
        w, b = -w, -b
    w_unit = w / np.linalg.norm(w)
    proj = X @ w_unit
    return {
        "w_unit": w_unit.astype(np.float32),
        "intercept": b,
        "layer": layer,
        "sigma_proj": float(proj.std()),
        "mean_proj_act1": float(proj[y].mean()) if y.any() else None,
        "mean_proj_act0": float(proj[~y].mean()) if (~y).any() else None,
        "train_acc": float(clf.score(X, y)),
        "n_samples": int(len(y)),
    }


def save_direction(results_dir, model_key, env_name, X_by_seed, y_by_seed, best_layer):
    d = fit_direction(X_by_seed, y_by_seed, best_layer)
    out = results_dir / f"direction_{model_key}_{env_name}.npz"
    np.savez(out, **{k: (np.nan if v is None else v) for k, v in d.items()})
    print(f"  direction: layer {d['layer']}  train_acc={d['train_acc']:.3f}  "
          f"sigma_proj={d['sigma_proj']:.3f}  -> {out.name}")
    return out


def plot_curve(curve, curve_bal, base, model_key, env_name, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(k for k, v in curve.items() if v is not None)
    accs = [curve[k] for k in layers]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, accs, "o-", color="tab:green", label="LOSO probe accuracy")
    layers_b = sorted(k for k, v in curve_bal.items() if v is not None)
    if layers_b:
        ax.plot(layers_b, [curve_bal[k] for k in layers_b], "s--",
                color="tab:olive", ms=4, lw=1,
                label="balanced (macro-avg over classes)")
    ax.axhline(base, color="gray", ls="--", label=f"majority baseline {base:.2f}")
    ax.axhline(0.5, color="lightgray", ls=":", lw=1,
               label="balanced chance 0.50")
    ax.set_xlabel("layer k (0 = embedding)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"P1 linear probe | {model_key} | {env_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_combo(residuals_dir, results_dir, plots_dir, model_key, env_name, save_weights=False):
    X_by_seed, y_by_seed = load_pool(residuals_dir, model_key, env_name)
    if not X_by_seed:
        print(f"[skip] no residuals for {model_key}/{env_name} in {residuals_dir}")
        return None
    if len(X_by_seed) < 2:
        print(f"[skip] {model_key}/{env_name}: {len(X_by_seed)} seed(s), LOSO needs >= 2")
        return None
    n_layers = next(iter(X_by_seed.values())).shape[1]
    n_total = sum(len(y) for y in y_by_seed.values())
    base = majority_baseline(y_by_seed)
    print(f"\n=== P1 probe: {model_key} | {env_name} | {n_total} rounds, "
          f"{len(X_by_seed)} seeds | majority baseline {base:.2f} ===")

    curve, curve_bal = {}, {}
    for layer in tqdm(range(n_layers), desc=f"{model_key}/{env_name}", unit="layer", leave=False):
        curve[layer], curve_bal[layer] = loso_accuracy(X_by_seed, y_by_seed, layer)
    for layer, acc in curve.items():
        if acc is not None:
            b = curve_bal[layer]
            print(f"  layer {layer:2d}  acc={acc:.3f}  bal={b:.3f}  {'#' * int(acc * 40)}"
                  if b is not None else
                  f"  layer {layer:2d}  acc={acc:.3f}  {'#' * int(acc * 40)}")

    valid = {k: v for k, v in curve.items() if v is not None}
    # best layer selected on raw accuracy, unchanged from P1_v3, so best_layer
    # (and every downstream direction/steering artifact) stays comparable;
    # balanced accuracy is reported alongside at every layer
    best_layer = max(valid, key=valid.get) if valid else None
    record = {
        "model": model_key,
        "env": env_name,
        "n_rounds_used": int(n_total),
        "n_seeds": len(X_by_seed),
        "ambig_eps": AMBIG_EPS,
        "probe_C": PROBE_C,
        "majority_baseline": base,
        "balanced_baseline": 0.5,
        "best_layer": best_layer,
        "best_accuracy": valid.get(best_layer),
        "best_accuracy_bal": curve_bal.get(best_layer),
        "final_layer_accuracy": valid.get(n_layers - 1),
        "margin_over_baseline": (valid[best_layer] - base) if best_layer is not None else None,
        "layer_accuracy": curve,
        "layer_accuracy_bal": curve_bal,
    }
    out_json = results_dir / f"probe_{model_key}_{env_name}.json"
    out_json.write_text(json.dumps(record, indent=2))
    if not valid:
        # every LOSO fold had a single-class training set (labels constant per
        # seed and too few seeds to mix classes) -- probe undefined, not zero
        record["note"] = "LOSO undefined: single-class training folds; add seeds"
        out_json.write_text(json.dumps(record, indent=2))
        print(f"  [undefined] all training folds single-class -- need more seeds")
        return record
    plot_curve(curve, curve_bal, base, model_key, env_name,
               plots_dir / f"probe_{model_key}_{env_name}.png")
    bal_str = (f" bal={record['best_accuracy_bal']:.3f}"
               if record['best_accuracy_bal'] is not None else "")
    print(f"  best layer {best_layer} acc={record['best_accuracy']:.3f}{bal_str} "
          f"(+{record['margin_over_baseline']:.3f} over baseline) -> {out_json.name}")
    if save_weights:
        save_direction(results_dir, model_key, env_name, X_by_seed, y_by_seed, best_layer)
    return record


def weights_only_combo(residuals_dir, results_dir, model_key, env_name):
    """Fast path: reuse best_layer from an existing probe_{model}_{env}.json and
    fit + save only the steering direction (no LOSO curve recompute)."""
    probe_json = results_dir / f"probe_{model_key}_{env_name}.json"
    if not probe_json.exists():
        print(f"[skip] {model_key}/{env_name}: no {probe_json.name} (run full probe first)")
        return None
    best_layer = json.loads(probe_json.read_text()).get("best_layer")
    if best_layer is None:
        print(f"[skip] {model_key}/{env_name}: best_layer undefined")
        return None
    X_by_seed, y_by_seed = load_pool(residuals_dir, model_key, env_name)
    if not X_by_seed:
        print(f"[skip] no residuals for {model_key}/{env_name} in {residuals_dir}")
        return None
    print(f"=== direction: {model_key} | {env_name} | best_layer={best_layer} ===")
    return save_direction(results_dir, model_key, env_name, X_by_seed, y_by_seed, int(best_layer))


def main():
    ap = argparse.ArgumentParser(description="P1 residual-stream linear probes")
    ap.add_argument("--residuals-dir", default="residuals/P2_v1",
                    help="folder with {model}_{env}_seed{N}.npy + _labels.json")
    ap.add_argument("--tag", default="P1_v1",
                    help="output subfolder name under results/ and plots/")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model keys")
    ap.add_argument("--envs", nargs="*", default=None, help="subset of env names")
    ap.add_argument("--save-weights", action="store_true",
                    help="also refit the best-layer probe on all data and save the "
                         "steering direction to direction_{model}_{env}.npz")
    ap.add_argument("--weights-only", action="store_true",
                    help="skip the LOSO curve; reuse best_layer from existing probe "
                         "JSONs and only fit + save the steering directions (fast)")
    args = ap.parse_args()

    residuals_dir = (HERE / args.residuals_dir).resolve()
    results_dir = HERE / "results" / args.tag
    plots_dir = HERE / "plots" / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    combos = set()
    for npy in residuals_dir.glob("*_seed*.npy"):
        model_key, env_name = npy.stem.rsplit("_seed", 1)[0].split("_", 1)
        if args.models and model_key not in args.models:
            continue
        if args.envs and env_name not in args.envs:
            continue
        combos.add((model_key, env_name))
    if not combos:
        sys.exit(f"no residuals matching filters in {residuals_dir}")

    if args.weights_only:
        for model_key, env_name in sorted(combos):
            weights_only_combo(residuals_dir, results_dir, model_key, env_name)
        return

    summary = []
    for model_key, env_name in sorted(combos):
        rec = run_combo(residuals_dir, results_dir, plots_dir, model_key, env_name,
                        save_weights=args.save_weights)
        if rec:
            summary.append({k: rec[k] for k in
                            ("model", "env", "majority_baseline", "best_layer",
                             "best_accuracy", "best_accuracy_bal",
                             "final_layer_accuracy",
                             "margin_over_baseline", "n_rounds_used", "n_seeds")})
    (results_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary ({len(summary)} combos) -> {results_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
