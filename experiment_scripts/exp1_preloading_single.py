# exp1_preloading_single.py
# Usage: python3 exp1_preloading_single.py <layer> <model_path> <out_dir> <norms_csv> <seed_start> <n_seeds>
#
# Experiment 1: are upcoming animals workspace-active before they are emitted?
# Builds J-lens probe vectors at `layer`, runs `n_seeds` sampled animal-fluency
# rollouts and, per seed, records:
#   * percentile rank of the about-to-be-emitted animal among not-yet-emitted
#     probes at the pre-emission position (null = 0.5);
#   * lead-time curve: z-scored activation of each animal up to LAG_WINDOW
#     positions before its emission, plus a symmetric peri-emission window;
#   * category-label tracking (LABEL_TARGETS): activation of category words
#     (e.g. "water", "farm") while clustering in vs. outside their category, and
#     around switches into that category.
#
# If env var JLENS_PT points to a precomputed jacobian-lens checkpoint
# (https://github.com/anthropics/jacobian-lens), probes are read out through that
# lens; otherwise J-lens vectors are estimated from a corpus (JLENS_CORPUS).
#
# Output: <out_dir>/exp1_layer<LL>_seed<start>-<end>.json (+ probe vectors as .npy)
import sys, os, json, warnings, numpy as np, torch
from collections import defaultdict
from scipy.stats import wilcoxon, linregress, spearmanr
import jlens_common as J

LAG_WINDOW = 20   # one-sided lead-time window; d=0 is the position before emission
PERI_LAG_BACK = 20   # symmetric peri-event window; lag 0 is the event's own position
PERI_LAG_FWD = 20
TEMP, TOP_P = 0.9, 0.95

# (label word, target category) pairs read out passively. Label words are single
# tokens on all tokenizers used, are not animals in the norms, and do not share a
# leading token. Categories were enabled only if switches into them occur often
# enough (switch-into events per 100 rollouts, Gemma-9B / Llama-3.1-8B / Qwen2.5-7B,
# noted where measured).
LABEL_TARGETS = [
    ("water", "water"),
    ("sea", "water"),
    ("farm", "farm"),
    ("domestic", "farm"),
    ("domestic", "pets"),
    ("pets", "pets"),
    ("fly", "birdsBats"),
    ("air", "birdsBats"),
    ("savannah", "africa"),
    ("africa", "africa"),
    ("crawl", "insectsSpiders"),
    ("insect", "insectsSpiders"),
    ("australian", "australia"),
    ("asia", "asiaEurasia"),                                            # 106 / 68 / 64
    ("saddle", "beastsOfBurden"), ("burden", "beastsOfBurden"),         # 64 / 24 / 12
    ("fur", "furWool"), ("wool", "furWool"),                            # 35 / 78 / 52
    ("herd", "bovineDeer"), ("horns", "bovineDeer"),                    # 45 / 86 / 34
    ("rodents", "rodentsRabbits"),                                      # 48 / 70 / 25
]


def build_label_candidates(tok):
    """Token id per unique label word (leading token if multi-token), plus the list
    of multi-token words."""
    label_words = sorted({label for label, _ in LABEL_TARGETS})
    cand_words, cand_ids, multi_token_words = [], [], []
    for w in label_words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if not ids:
            raise ValueError(f"label word {w!r} encoded to zero tokens -- check the tokenizer")
        if len(ids) > 1:
            multi_token_words.append(w)
        cand_words.append(w)
        cand_ids.append(ids[0])
    return cand_words, cand_ids, multi_token_words


def find_label_token_collisions(label_words, label_cand_ids):
    """{token_id: [label words]} for token ids shared by 2+ labels (they would get
    identical probe directions). Empty dict is the clean case."""
    by_id = defaultdict(list)
    for w, t in zip(label_words, label_cand_ids):
        by_id[int(t)].append(w)
    return {t: ws for t, ws in by_id.items() if len(ws) > 1}


def compute_V(model, tok, layer, cand_ids, out_dir, layer_tag, jlens_pt):
    if jlens_pt:
        ckpt = J.load_jlens_pt(jlens_pt)
        cand_ids_t = torch.as_tensor(cand_ids, device=J.DEVICE)
        W_U, _ = J.get_unembed_and_norm(model)
        V = torch.nn.functional.normalize(
            W_U[cand_ids_t].float() @ ckpt["J"][layer].to(J.DEVICE, torch.float32), dim=1)
        J_for_lens = ckpt["J"]
    else:
        corpus = J.load_corpus(120, os.environ.get("JLENS_CORPUS"))
        V = J.compute_jlens_vectors(model, tok, cand_ids, layer, corpus)   # [C, D]
        J_for_lens = None
    np.save(os.path.join(out_dir, f"jlens_V_layer{layer_tag}.npy"), V.cpu().numpy())
    return V, J_for_lens


def run_one_seed(model, tok, layer, cand_ids, cand_words, cidx, word_of_id,
                  V, J_for_lens, jlens_pt, seed,
                  V_label=None, label_words=None, label_cand_ids=None, word2cats=None):
    full_ids, prompt_len = J.fluency_generate(model, tok, do_sample=True,
                                               temperature=TEMP, top_p=TOP_P, seed=seed)

    if jlens_pt:
        H_raw = J.read_activations(model, tok, full_ids, layer, normalize=False)
        A = J.lens_forward_logits(model, H_raw, layer, J_for_lens, cand_ids).cpu().numpy()
    else:
        H = J.read_activations(model, tok, full_ids, layer)               # [S, D]
        A = (H @ V.T).cpu().numpy()                                        # [S, C]
    gen = slice(prompt_len, full_ids.shape[0])
    Az = (A - A[gen].mean(0, keepdims=True)) / (A[gen].std(0, keepdims=True) + 1e-6)

    # first emission of each probe word
    gen_ids = full_ids[prompt_len:].tolist()
    emissions, seen = [], set()
    for off, tid in enumerate(gen_ids):
        w = word_of_id.get(tid)
        if w and w not in seen:
            emissions.append((prompt_len + off, w)); seen.add(w)

    # percentile rank of the upcoming word among not-yet-emitted probes
    already, pct = set(), []
    for e_abs, w in emissions:
        src = e_abs - 1
        if src < prompt_len:
            already.add(w); continue
        pool = [cidx[x] for x in cand_words if x not in already]
        if len(pool) >= 3:
            pct.append(float((A[src, pool] < A[src, cidx[w]]).mean()))
        already.add(w)
    pct = np.array(pct)

    leads = np.full((len(emissions), LAG_WINDOW + 1), np.nan)
    for r, (e_abs, w) in enumerate(emissions):
        for d in range(LAG_WINDOW + 1):
            p = e_abs - 1 - d
            if p >= prompt_len:
                leads[r, d] = Az[p, cidx[w]]
    emission_words = [w for _, w in emissions]

    S = full_ids.shape[0]
    peri_width = PERI_LAG_BACK + PERI_LAG_FWD + 1
    peri = np.full((len(emissions), peri_width), np.nan)
    for r, (e_abs, w) in enumerate(emissions):
        for i, d in enumerate(range(-PERI_LAG_BACK, PERI_LAG_FWD + 1)):
            p = e_abs + d
            if prompt_len <= p < S:
                peri[r, i] = Az[p, cidx[w]]

    lag_idx = np.broadcast_to(np.arange(LAG_WINDOW + 1), leads.shape)
    mask = ~np.isnan(leads)
    lags_flat = lag_idx[mask].astype(float)
    vals_flat = leads[mask]
    if lags_flat.size >= 3:
        fit = linregress(lags_flat, vals_flat)
        rho, rho_p = spearmanr(lags_flat, vals_flat)
        decay = {
            "lag_decay_slope": float(fit.slope),
            "lag_decay_intercept": float(fit.intercept),
            "lag_decay_r": float(fit.rvalue),
            "lag_decay_p": float(fit.pvalue),
            "lag_decay_spearman_rho": float(rho),
            "lag_decay_spearman_p": float(rho_p),
            "lag_decay_n": int(lags_flat.size),
        }
    else:
        decay = {"lag_decay_slope": None, "lag_decay_intercept": None,
                  "lag_decay_r": None, "lag_decay_p": None,
                  "lag_decay_spearman_rho": None, "lag_decay_spearman_p": None,
                  "lag_decay_n": int(lags_flat.size)}

    gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)

    # Category-label tracking (empty dicts when no norms file was given)
    label_results = {}
    label_switch_groups = {}
    if V_label is not None and word2cats is not None:
        import jlens_norms as N
        if jlens_pt:
            A_label = J.lens_forward_logits(model, H_raw, layer, J_for_lens, label_cand_ids).cpu().numpy()
        else:
            A_label = (H @ V_label.T).cpu().numpy()
        Az_label = (A_label - A_label[gen].mean(0, keepdims=True)) / (A_label[gen].std(0, keepdims=True) + 1e-6)
        label_idx = {w: i for i, w in enumerate(label_words)}

        labels_by_target = defaultdict(list)
        for label, target_cat in LABEL_TARGETS:
            labels_by_target[target_cat].append(label)
        pair_keys = [f"{label}->{target_cat}" for label, target_cat in LABEL_TARGETS]

        tracked = N.build_tracked_sequence(gen_text, gen_ids, prompt_len, word_of_id)

        # (1) Clustering in target vs. other: for every tracked item (repeats
        # included), file each label's activation at that item under "in_target" if
        # the item belongs to the label's target category, else "in_other".
        # Items with no known category are excluded from both.
        cluster_acc = {pk: {"in_target": [], "in_other": []} for pk in pair_keys}
        for pos, w, _gap in tracked:
            cats_w = word2cats.get(w)
            if not cats_w:
                continue
            for label, target_cat in LABEL_TARGETS:
                z = float(Az_label[pos, label_idx[label]])
                pk = f"{label}->{target_cat}"
                (cluster_acc[pk]["in_target"] if target_cat in cats_w
                 else cluster_acc[pk]["in_other"]).append(z)

        # (2) Switch into target: peri-event curve locked to transitions where item
        # n+1 is in target_cat and item n is not. Non-adjacent (gapped) pairs skipped.
        width = PERI_LAG_BACK + PERI_LAG_FWD + 1
        lag_acc = {pk: [] for pk in pair_keys}
        S = full_ids.shape[0]
        for n in range(len(tracked) - 1):
            _, w, _ = tracked[n]
            next_abs, w_next, gap_next = tracked[n + 1]
            if gap_next:
                continue
            cats_w, cats_next = word2cats.get(w, set()), word2cats.get(w_next, set())
            for target_cat, labels_here in labels_by_target.items():
                if target_cat not in cats_next or target_cat in cats_w:
                    continue
                for label in labels_here:
                    label_lag = np.full(width, np.nan)
                    for i, d in enumerate(range(-PERI_LAG_BACK, PERI_LAG_FWD + 1)):
                        p = next_abs + d
                        if prompt_len <= p < S:
                            label_lag[i] = Az_label[p, label_idx[label]]
                    lag_acc[f"{label}->{target_cat}"].append(label_lag)

        for pk in pair_keys:
            tgt, oth = cluster_acc[pk]["in_target"], cluster_acc[pk]["in_other"]
            lags = lag_acc[pk]
            if lags:
                stacked = np.array(lags, dtype=float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    lag_mean = np.nanmean(stacked, axis=0)
                lag_mean = [None if np.isnan(x) else float(x) for x in lag_mean]
            else:
                lag_mean = [None] * width
            label_results[pk] = {
                "n_in_target": len(tgt), "mean_z_in_target": float(np.mean(tgt)) if tgt else None,
                "n_in_other": len(oth), "mean_z_in_other": float(np.mean(oth)) if oth else None,
                "n_switch_into_events": len(lags),
                "lag_at_switch_into": lag_mean,
            }

        # (3) Control for (2): at every clean switch (no shared category), each
        # label's peri-event curve is filed by whether its target categories overlap
        # the destination ("into"), the source ("from"), or neither ("unrelated").
        # Saved per label word: {label: {group: {"n", "lag"}}}.
        label_targets_map = defaultdict(set)
        for label, target_cat in LABEL_TARGETS:
            label_targets_map[label].add(target_cat)
        grp_acc = {lab: {"into": [], "from": [], "unrelated": []} for lab in label_words}
        for n in range(len(tracked) - 1):
            _, w, _ = tracked[n]
            next_abs, w_next, gap_next = tracked[n + 1]
            if gap_next or not N.is_switch(w, w_next, word2cats):
                continue
            cats_w, cats_next = word2cats.get(w, set()), word2cats.get(w_next, set())
            win = np.full((width, len(label_words)), np.nan)
            for i, d in enumerate(range(-PERI_LAG_BACK, PERI_LAG_FWD + 1)):
                p = next_abs + d
                if prompt_len <= p < S:
                    win[i, :] = Az_label[p, :]
            for lab in label_words:
                tg = label_targets_map[lab]
                col = win[:, label_idx[lab]]
                if tg & cats_next:
                    grp_acc[lab]["into"].append(col)
                if tg & cats_w:
                    grp_acc[lab]["from"].append(col)
                if not (tg & (cats_w | cats_next)):
                    grp_acc[lab]["unrelated"].append(col)
        for lab in label_words:
            label_switch_groups[lab] = {}
            for grp, curves in grp_acc[lab].items():
                if curves:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        m = np.nanmean(np.array(curves, dtype=float), axis=0)
                    label_switch_groups[lab][grp] = {"n": len(curves),
                                                     "lag": [None if np.isnan(x) else float(x) for x in m]}
                else:
                    label_switch_groups[lab][grp] = {"n": 0, "lag": [None] * width}

    return {
        "seed": seed,
        "n_emissions": int(len(pct)),
        "median_percentile": float(np.median(pct)) if len(pct) else None,
        "wilcoxon_p": (float(wilcoxon(pct - 0.5, alternative="greater")[1])
                       if len(pct) >= 6 else None),
        "lead_time_mean": np.nanmean(leads, axis=0).tolist(),
        "lead_time_se": (np.nanstd(leads, axis=0)
                         / np.sqrt(np.maximum(np.sum(~np.isnan(leads), axis=0), 1))).tolist(),
        "lead_time_words": emission_words,
        "lead_time_raw": leads.tolist(),   # [n_emissions, LAG_WINDOW+1]
        **decay,
        "peri_emission_mean": np.nanmean(peri, axis=0).tolist(),
        "peri_emission_se": (np.nanstd(peri, axis=0)
                             / np.sqrt(np.maximum(np.sum(~np.isnan(peri), axis=0), 1))).tolist(),
        "peri_emission_raw": peri.tolist(),   # [n_emissions, PERI_LAG_BACK+PERI_LAG_FWD+1]
        "label_results": label_results,             # keyed "{label}->{target_cat}"
        "label_switch_groups": label_switch_groups,
        "generated": gen_text,
    }


def main():
    layer      = int(sys.argv[1])
    model_path = sys.argv[2]
    out_dir    = sys.argv[3]
    norms_csv  = sys.argv[4] if len(sys.argv) > 4 else None
    seed_start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    n_seeds    = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    os.makedirs(out_dir, exist_ok=True)

    model, tok = J.load_model(model_path)
    import jlens_norms as N

    word2cats = None
    if norms_csv:
        norms = N.load_troyer_norms(norms_csv, tok)
        cand = norms["probe_ids"]
        word2cats = norms["word2cats"]
    else:
        cand = J.single_token_candidates(tok)

    cand_words = list(cand.keys())
    cand_ids = [cand[w] for w in cand_words]
    cidx = {w: k for k, w in enumerate(cand_words)}
    word_of_id = N.build_word_of_id(tok, cand)
    print(f"layer {layer}: {len(cand_words)} probes, seeds {seed_start}-{seed_start + n_seeds - 1}",
          flush=True)

    jlens_pt = os.environ.get("JLENS_PT")
    seed_end = seed_start + n_seeds - 1
    layer_tag = f"{layer:02d}_seed{seed_start:03d}-{seed_end:03d}"
    V, J_for_lens = compute_V(model, tok, layer, cand_ids, out_dir, layer_tag, jlens_pt)

    # Category-label probes (need a norms file to classify items by category)
    label_words = label_cand_ids = V_label = None
    multi_token_labels = []
    label_token_collisions = {}
    if word2cats is not None:
        label_words, label_cand_ids, multi_token_labels = build_label_candidates(tok)
        label_token_collisions = find_label_token_collisions(label_words, label_cand_ids)
        if label_token_collisions:
            print(f"WARNING: label words share a token id -- IDENTICAL directions, curves will be duplicates: "
                  f"{label_token_collisions}", flush=True)
        if multi_token_labels:
            print(f"NOTE: {multi_token_labels} are multi-token in this tokenizer -- using each "
                  f"word's LEADING token as a proxy", flush=True)
        V_label = J.get_probe_vectors(model, tok, layer, label_cand_ids, jlens_pt=jlens_pt)
        np.save(os.path.join(out_dir, f"jlens_Vlabel_layer{layer_tag}.npy"), V_label.cpu().numpy())

    records = []
    for seed in range(seed_start, seed_start + n_seeds):
        print(f"  seed {seed} ...", flush=True)
        records.append(run_one_seed(model, tok, layer, cand_ids, cand_words, cidx, word_of_id,
                                     V, J_for_lens, jlens_pt, seed,
                                     V_label=V_label, label_words=label_words,
                                     label_cand_ids=label_cand_ids, word2cats=word2cats))

    out = {
        "layer": layer,
        "n_probes": len(cand_words),
        "probe_words": cand_words,          # row order matches jlens_V_layer*.npy
        "used_precomputed_lens": bool(jlens_pt),
        "jlens_pt": jlens_pt,
        "label_targets": LABEL_TARGETS,
        "label_words": label_words,         # row order matches jlens_Vlabel_layer*.npy
        "multi_token_labels": multi_token_labels,
        "label_token_collisions": label_token_collisions,
        "peri_lag_back": PERI_LAG_BACK, "peri_lag_fwd": PERI_LAG_FWD,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "records": records,
    }
    with open(os.path.join(out_dir, f"exp1_layer{layer_tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    medians = [r["median_percentile"] for r in records if r["median_percentile"] is not None]
    slopes = [r["lag_decay_slope"] for r in records if r["lag_decay_slope"] is not None]
    print(json.dumps({
        "layer": layer, "n_seeds": n_seeds,
        "mean_median_percentile": float(np.mean(medians)) if medians else None,
        "mean_lag_decay_slope": float(np.mean(slopes)) if slopes else None,
    }, indent=2))


if __name__ == "__main__":
    main()
