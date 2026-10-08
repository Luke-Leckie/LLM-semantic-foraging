# exp5_analysis_single.py
# Usage: python3 exp5_analysis_single.py <results_exp5_dir> <out_dir> <space> <layer> <position> <norms_csv> [n_permutations] [n_pca]
#   space:    "residual" or "mlp"
#   layer:    a layer captured by exp5_switch_neurons_full_single.py
#   position: "pre" or "emit"
#
# Experiment 5 (analysis, CPU only): for one (space, layer, position), reads the
# captured activations and tests for a category-general switch signal:
#   * screening: per-dimension point-biserial r with switch, and how consistent its
#     sign is across source categories;
#   * probe: PCA (n_pca components) + L2 logistic regression, evaluated with
#     leave-category-out cross-validation (AUC) against a label-permutation null.
#     Animals can belong to several categories, so an event is held out of
#     training whenever ANY of its word's categories is in the test fold.
# For position "pre", also fits the probe on all valid events and saves the
# coefficient vector projected back to activation space (unit norm) as the
# generic switch direction (the residual one is the steering direction in exp6).
#
# Output: <out_dir>/exp5analysis_<space>_<position>_layer<LL>.json
#         <out_dir>/generic_{switch,mlp}_direction_layer<LL>.npy (position "pre" only)
import sys, os, json, glob
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import jlens_norms as N

MIN_GROUP_N = 10


def load_exp5(d):
    """Merge all exp5 chunk files (meta JSON + npz) in seed order, keeping array rows
    aligned with the events DataFrame."""
    meta_fps = sorted(glob.glob(os.path.join(d, "exp5_seed*_meta.json")),
                       key=lambda fp: json.load(open(fp))["seed_start"])
    all_events, arrays = [], {}
    for meta_fp in meta_fps:
        meta = json.load(open(meta_fp))
        all_events.extend(meta["events"])
        npz_fp = meta_fp.replace("_meta.json", ".npz")
        with np.load(npz_fp) as npz:
            for key in npz.files:
                arrays.setdefault(key, []).append(npz[key])
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    return pd.DataFrame(all_events), arrays


def pointbiserial_vec(switch, arr):
    """Vectorized point-biserial (= Pearson) correlation between binary `switch` [n]
    and each column of `arr` [n, D]."""
    x = switch - switch.mean()
    y = arr - arr.mean(axis=0, keepdims=True)
    num = x @ y
    den = np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.nan_to_num(r, nan=0.0)


def screen_neurons(events_df, arr, min_group_n=MIN_GROUP_N, top_n=50):
    """Per-dimension pooled point-biserial r vs. switch, plus the fraction of source
    categories in which r has the same sign. Returns the top `top_n` dims by |r|."""
    switch = events_df["switch"].values.astype(float)
    r_pooled = pointbiserial_vec(switch, arr)

    cats = events_df["src_cat"].values
    cat_counts = pd.Series(cats).value_counts()
    valid_cats = cat_counts[cat_counts >= min_group_n].index

    sign_matches = np.zeros(arr.shape[1])
    n_valid_groups = 0
    for c in valid_cats:
        mask = cats == c
        sub_switch = switch[mask]
        if len(np.unique(sub_switch)) < 2:
            continue
        n_valid_groups += 1
        r_sub = pointbiserial_vec(sub_switch, arr[mask])
        sign_matches += (np.sign(r_sub) == np.sign(r_pooled)) & (r_pooled != 0)
    consistency = sign_matches / n_valid_groups if n_valid_groups else np.full(arr.shape[1], np.nan)

    order = np.argsort(-np.abs(r_pooled))[:top_n]
    return {
        "n_groups": int(n_valid_groups),
        "top_dims": order.tolist(),
        "top_r": [float(v) for v in r_pooled[order]],
        "top_consistency": [float(v) for v in consistency[order]],
    }


def _multi_label_category_counts(words, word2cats):
    """({category: n_events}, per-event category sets). An event counts toward every
    category its word belongs to."""
    cats_per_event = [word2cats.get(w, set()) for w in words]
    cat_counts = Counter()
    for cats in cats_per_event:
        for c in cats:
            cat_counts[c] += 1
    return cat_counts, cats_per_event


def multi_label_category_folds(all_categories, n_splits, seed=0):
    """Shuffle categories (deterministically) into `n_splits` disjoint held-out sets."""
    cats = sorted(all_categories)
    order = np.random.RandomState(seed).permutation(len(cats))
    folds = [set() for _ in range(n_splits)]
    for i, idx in enumerate(order):
        folds[i % n_splits].add(cats[idx])
    return folds


def split_multi_label(cats_per_event, held_out_cats):
    """Test = events whose word has any category in held_out_cats; train = all others."""
    train_idx, test_idx = [], []
    for i, cats in enumerate(cats_per_event):
        (test_idx if cats & held_out_cats else train_idx).append(i)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def held_out_probe_pca(events_df, arr, word2cats, min_group_n=MIN_GROUP_N,
                        n_permutations=100, n_pca=30, seed=0):
    """Leave-category-out PCA + logistic-regression probe (PCA refit on each training
    fold), with a label-permutation null. Only categories with >= min_group_n events
    are used."""
    switch = events_df["switch"].values.astype(int)
    words = events_df["word"].values
    cat_counts, cats_per_event = _multi_label_category_counts(words, word2cats)
    valid_cats = {c for c, n in cat_counts.items() if n >= min_group_n}
    valid_mask = np.array([bool(cats & valid_cats) for cats in cats_per_event])
    X, y = arr[valid_mask], switch[valid_mask]
    cats_valid = [cats for cats, keep in zip(cats_per_event, valid_mask) if keep]

    n_splits = min(5, len(valid_cats))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return None

    # folds are fixed; permutations only shuffle labels
    folds = multi_label_category_folds(valid_cats, n_splits, seed=0)
    fold_splits = [split_multi_label(cats_valid, held_out) for held_out in folds]

    def cv_auc(X, y):
        """(mean AUC, per-fold AUCs) over folds with both classes in train and test."""
        aucs = []
        for train_idx, test_idx in fold_splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            pca = PCA(n_components=min(n_pca, len(train_idx) - 1), svd_solver="randomized", random_state=0)
            Xtr = pca.fit_transform(X[train_idx])
            Xte = pca.transform(X[test_idx])
            clf = LogisticRegression(max_iter=200, C=0.1, solver="liblinear")
            clf.fit(Xtr, y[train_idx])
            proba = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[test_idx], proba))
        return (np.mean(aucs) if aucs else np.nan), aucs

    obs_auc, obs_fold_aucs = cv_auc(X, y)
    rng = np.random.RandomState(seed)
    null_aucs = np.array([cv_auc(X, rng.permutation(y))[0] for _ in range(n_permutations)])
    null_aucs = null_aucs[~np.isnan(null_aucs)]
    p_perm = (np.sum(null_aucs >= obs_auc) + 1) / (len(null_aucs) + 1) if len(null_aucs) else np.nan

    return {"held_out_auc": float(obs_auc), "null_mean": float(null_aucs.mean()) if len(null_aucs) else None,
            "null_std": float(null_aucs.std()) if len(null_aucs) else None, "p_perm": float(p_perm),
            "n_permutations": int(len(null_aucs)), "n_valid_events": int(valid_mask.sum()),
            "fold_aucs": [float(a) for a in obs_fold_aucs]}


def fit_generic_direction(events_df, arr, word2cats, min_group_n=MIN_GROUP_N, n_pca=30):
    """Fit PCA + logistic regression on all valid events and return the coefficient
    vector projected back to activation space, d = W beta, normalised to unit length.
    Generalisation is assessed by held_out_probe_pca, not by this fit."""
    switch = events_df["switch"].values.astype(int)
    words = events_df["word"].values
    cat_counts, cats_per_event = _multi_label_category_counts(words, word2cats)
    valid_cats = {c for c, n in cat_counts.items() if n >= min_group_n}
    valid_mask = np.array([bool(cats & valid_cats) for cats in cats_per_event])
    X, y = arr[valid_mask], switch[valid_mask]
    if len(np.unique(y)) < 2:
        return None
    pca = PCA(n_components=min(n_pca, X.shape[0] - 1), svd_solver="randomized", random_state=0)
    Xp = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=0.1, solver="liblinear")
    clf.fit(Xp, y)
    direction = pca.components_.T @ clf.coef_[0]          # [D] <- [D, n_pca] @ [n_pca]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction.astype(np.float32)


def main():
    results_dir = sys.argv[1]
    out_dir     = sys.argv[2]
    space       = sys.argv[3]
    layer       = int(sys.argv[4])
    position    = sys.argv[5]
    norms_csv   = sys.argv[6]
    n_permutations = int(sys.argv[7]) if len(sys.argv) > 7 else 100
    n_pca          = int(sys.argv[8]) if len(sys.argv) > 8 else 30
    os.makedirs(out_dir, exist_ok=True)

    assert space in ("residual", "mlp")
    assert position in ("pre", "emit")

    word2cats = N.load_troyer_norms(norms_csv, N.WhitespaceTokStub(), verbose=False)["word2cats"]

    events_df, arrays = load_exp5(results_dir)
    key = f"{space}_{position}_layer{layer:02d}"
    if key not in arrays:
        raise KeyError(f"{key} not found in captured exp5 arrays: {sorted(arrays.keys())}")
    arr = arrays[key]
    print(f"{key}: {len(events_df)} events, activation shape {arr.shape}", flush=True)

    screening = screen_neurons(events_df, arr)
    probe = held_out_probe_pca(events_df, arr, word2cats, n_permutations=n_permutations, n_pca=n_pca)

    out = {"space": space, "layer": layer, "position": position,
           "n_pca": n_pca, "n_permutations_requested": n_permutations,
           "n_events_total": len(events_df), "screening": screening, "probe": probe}

    tag = f"exp5analysis_{space}_{position}_layer{layer:02d}"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    if position == "pre":
        direction = fit_generic_direction(events_df, arr, word2cats, n_pca=n_pca)
        if direction is not None:
            tag = "generic_switch_direction" if space == "residual" else "generic_mlp_direction"
            np.save(os.path.join(out_dir, f"{tag}_layer{layer:02d}.npy"), direction)
            print(f"saved {tag}_layer{layer:02d}.npy (dim={direction.shape[0]})", flush=True)

    print(json.dumps({"space": space, "layer": layer, "position": position, "probe": probe}, indent=2))


if __name__ == "__main__":
    main()
