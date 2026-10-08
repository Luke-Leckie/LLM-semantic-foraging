# exp9_direction_loadings_single.py
# Usage: python3 exp9_direction_loadings_single.py <model_path> <out_dir> <norms_csv> <generic_dir> \
#            <seed_start> <n_seeds> [n_random=8] [spaces=residual,mlp] [capture_dir=auto]
#
# Experiment 9: how does the projection onto exp5's switch directions evolve as the
# model spends longer inside a category cluster, up to the switch? If the direction
# tracks an accumulating urge to leave the patch, loading should rise towards the switch.
#
# Directions: the unit-norm generic switch directions in <generic_dir>, for the "pre"
# position (the item before a decision; the steering direction in exp6) and, where
# available, the "emit" position (the item itself). Missing directions are fitted
# from the raw exp5 capture (<capture_dir>, by default derived from generic_dir's
# name) and saved into generic_dir.
#
# Loading: raw projection h . d of every generated token's activation onto each
# direction (residual: hidden_states[layer+1]; MLP: down_proj input). Activation
# norms are saved too so cosines can be formed downstream. Positive = more
# switch-like. Control: n_random random unit directions orthogonal to each real
# direction, drawn from a fixed seed so they are identical across seed chunks.
# Row 0 of every loading array is the real direction, rows 1..n_random the controls.
#
# Clusters: every recognised animal (single- or multi-token) is a tracked item. A
# cluster is a maximal run of adjacent items sharing a category with the previous
# one. It ends in a switch if the next adjacent item shares no category; it is
# censored if it ends at the end of generation or at an unrecognised item. For each
# generated token: cluster_id, tok_since_start (tokens since the cluster's first
# item) and tok_to_switch (tokens until the switch destination item, 0 at the
# destination itself; -1 if undefined or censored).
#
# Rollouts use the same prompt, sampling and seeds as exp5, without steering.
#
# Output (per seed chunk):
#   exp9_seed<S>-<E>.npz: seeds, prompt_len, gen_len; per-token cluster_id /
#     tok_since_start / tok_to_switch [n_seeds, T]; loadings res_L<NN>_<pos> /
#     mlp_L<NN>_<pos> [n_seeds, 1+n_random, T]; norms resnorm_L<NN> / mlpnorm_L<NN>.
#     Padded with -1 / NaN.
#   exp9_seed<S>-<E>_meta.json: per-seed text, items and clusters, run settings, and
#     a `canary` check: per residual layer, Cohen's d of the real direction's loading
#     at switch vs. stay items (should be clearly > 0 where exp5's probe AUC was
#     high), next to the same statistic for the random controls (~0).
import sys, os, re, json, glob
import numpy as np
import torch
import jlens_common as J
import jlens_norms as N
from exp5_analysis_single import fit_generic_direction
from fit_emit_directions import load_exp5_events, load_one_layer_array, layers_captured

TEMP, TOP_P = 0.9, 0.95
GEN_MAX_NEW_TOKENS = 220
RANDOM_SEED = 0          # fixed so random control directions are identical across chunks
N_RANDOM_DEFAULT = 8
PROMPT = ("Name as many different animals as you can, one after another, "
          "separated by commas. Just the list.")
POSITIONS = ("pre", "emit")


def load_directions(generic_dir, spaces):
    """{space: {position: {layer: unit-norm float32 vector}}} from the direction files in
    generic_dir ("pre": ..._direction_layer<NN>.npy, "emit": ..._direction_emit_layer<NN>.npy)."""
    patterns = {
        "residual": {"pre": "generic_switch_direction_layer*.npy",
                     "emit": "generic_switch_direction_emit_layer*.npy"},
        "mlp": {"pre": "generic_mlp_direction_layer*.npy",
                "emit": "generic_mlp_direction_emit_layer*.npy"},
    }
    out = {"residual": {"pre": {}, "emit": {}}, "mlp": {"pre": {}, "emit": {}}}
    for space in spaces:
        for position, pattern in patterns[space].items():
            for fp in sorted(glob.glob(os.path.join(generic_dir, pattern))):
                layer = int(re.search(r"layer(\d+)\.npy$", fp).group(1))
                out[space][position][layer] = np.load(fp).astype(np.float32)
    return out


def _default_capture_dir(generic_dir):
    """Raw exp5 capture directory for `generic_dir` by naming convention
    (results_exp5_full_analysis_2b -> results_exp5_full_2b); None if it doesn't apply."""
    parent, name = os.path.split(generic_dir.rstrip("/"))
    if "_full_analysis" not in name:
        return None
    return os.path.join(parent, name.replace("_full_analysis", "_full", 1))


def _fit_and_fill(generic_dir, capture_dir, dirs, norms_csv, space, tag, position, target_layers,
                  events_cache):
    """Fit and save directions for layers in `target_layers` missing from dirs[space][position]
    (updates `dirs` in place). `events_cache` lets the exp5 events be loaded only once."""
    missing = sorted(set(target_layers) - set(dirs[space][position]))
    if not missing:
        return
    if not events_cache:
        events_df, meta_fps = load_exp5_events(capture_dir)
        word2cats = N.load_troyer_norms(norms_csv, N.WhitespaceTokStub(), verbose=False)["word2cats"]
        print(f"fitting missing direction(s) from {capture_dir} ({len(events_df)} events)...", flush=True)
        events_cache.append((events_df, meta_fps, word2cats))
    events_df, meta_fps, word2cats = events_cache[0]
    infix = "" if position == "pre" else f"_{position}"
    for L in missing:
        out_fp = os.path.join(generic_dir, f"{tag}{infix}_layer{L:02d}.npy")
        if os.path.exists(out_fp):            # written by a concurrent job
            dirs[space][position][L] = np.load(out_fp).astype(np.float32)
            print(f"  {space}/{position} layer {L:2d}: {out_fp} already exists (written concurrently), loaded", flush=True)
            continue
        arr = load_one_layer_array(meta_fps, f"{space}_{position}_layer{L:02d}")
        if arr is None:
            continue                          # layer not captured
        direction = fit_generic_direction(events_df, arr, word2cats, n_pca=30)
        del arr
        if direction is None:
            print(f"  {space}/{position} layer {L:2d}: fit_generic_direction returned None (too few valid events), skipped", flush=True)
            continue
        np.save(out_fp, direction)
        dirs[space][position][L] = direction.astype(np.float32)
        print(f"  {space}/{position} layer {L:2d}: fit and saved {os.path.basename(out_fp)}", flush=True)


def fit_missing_directions(generic_dir, capture_dir, spaces, dirs, norms_csv):
    """Fit and save any direction files missing from `generic_dir` (updates `dirs` in place),
    using exp5_analysis_single.fit_generic_direction on the raw capture in `capture_dir`:
      1. "pre": every captured layer without a direction file;
      2. "emit": every layer that has a "pre" direction but no "emit" one.
    Layers that cannot be fitted are left without a direction. Concurrent jobs may refit the
    same layer; fits are deterministic and existing files are reused."""
    if not capture_dir or not os.path.isdir(capture_dir):
        return
    # list capture files to find which layers exist (all chunks capture the same layers)
    meta_fps_probe = sorted(glob.glob(os.path.join(capture_dir, "exp5_seed*_meta.json")))
    events_cache = []                         # events loaded lazily, at most once
    for space, tag in (("residual", "generic_switch_direction"), ("mlp", "generic_mlp_direction")):
        if space not in spaces:
            continue
        pre_target = layers_captured(meta_fps_probe, space, "pre")
        _fit_and_fill(generic_dir, capture_dir, dirs, norms_csv, space, tag, "pre", pre_target, events_cache)
        emit_target = list(dirs[space]["pre"])
        _fit_and_fill(generic_dir, capture_dir, dirs, norms_csv, space, tag, "emit", emit_target, events_cache)


def annotate_clusters(tracked, word2cats, prompt_len, gen_len):
    """(items, clusters, (cluster_id, tok_since_start, tok_to_switch)) from a tracked
    sequence [(abs_pos, word, gap_before), ...]; see the header for definitions. Token
    arrays cover the gen_len generated positions, -1 where undefined. A token belongs to
    the cluster of the latest item at or before it; tok_to_switch is set to 0 at each
    switch destination token."""
    items, clusters = [], []
    cur = None
    for pos, w, gap in tracked:
        starts_new = cur is None or gap or N.is_switch(items[-1]["word"], w, word2cats)
        if starts_new and cur is not None:
            # an adjacent item sharing no category is a switch; after a gap it is unknown
            cur["ends_in_switch"] = not gap
            cur["dest_pos"] = pos if not gap else None
            clusters.append(cur)
        if starts_new:
            cur = {"id": len(clusters), "start_pos": pos, "last_pos": pos, "n_items": 0,
                   "ends_in_switch": False, "dest_pos": None}
        cur["n_items"] += 1
        cur["last_pos"] = pos
        items.append({"pos": int(pos), "t": int(pos - prompt_len), "word": w,
                      "cats": sorted(word2cats.get(w, [])), "gap_before": bool(gap),
                      "cluster": cur["id"], "run_len": cur["n_items"],
                      "tok_since_start": int(pos - cur["start_pos"])})
    if cur is not None:
        clusters.append(cur)                      # last cluster: censored by end of generation
    for it in items:
        cl = clusters[it["cluster"]]
        it["cluster_n_items"] = cl["n_items"]
        it["ends_in_switch"] = bool(cl["ends_in_switch"])
        it["items_to_switch"] = (cl["n_items"] - it["run_len"]) if cl["ends_in_switch"] else None
        it["tok_to_switch"] = int(cl["dest_pos"] - it["pos"]) if cl["ends_in_switch"] else None

    cluster_id = np.full(gen_len, -1, np.int32)
    tok_since = np.full(gen_len, -1, np.int32)
    tok_to_sw = np.full(gen_len, -1, np.int32)
    if items:
        item_pos = np.array([it["pos"] for it in items])
        for t in range(gen_len):
            p = prompt_len + t
            k = int(np.searchsorted(item_pos, p, side="right")) - 1
            if k < 0:
                continue                          # before the first tracked item
            cl = clusters[items[k]["cluster"]]
            cluster_id[t] = cl["id"]
            tok_since[t] = p - cl["start_pos"]
            if cl["ends_in_switch"]:
                tok_to_sw[t] = cl["dest_pos"] - p
    for cl in clusters:
        if cl["ends_in_switch"]:
            dt = cl["dest_pos"] - prompt_len
            if 0 <= dt < gen_len:
                tok_to_sw[dt] = 0      # the switch destination token itself
    return items, clusters, (cluster_id, tok_since, tok_to_sw)


def project(acts, dmat):
    """acts [T, dim] (float32, on-device), dmat [1+R, dim] unit rows -> (loadings [1+R, T],
    norms [T]) as float32 numpy."""
    return (acts @ dmat.T).T.float().cpu().numpy(), acts.norm(dim=1).float().cpu().numpy()


def _cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else None


def _item_role(it, position):
    """Canary group of a tracked item at `position`, or None.
    "pre":  "switch" if the next item starts a new cluster via a switch, "stay" if the
            next item is in the same cluster.
    "emit": "switch" if the item is the first of a (non-initial, non-gapped) cluster,
            i.e. a switch destination; "stay" if it continues a cluster."""
    if position == "pre":
        if it["ends_in_switch"] and it["items_to_switch"] == 0:
            return "switch"
        if it["run_len"] < it["cluster_n_items"]:
            return "stay"
        return None
    else:  # "emit"
        if it["run_len"] == 1 and it["cluster"] > 0 and not it["gap_before"]:
            return "switch"
        if it["run_len"] > 1:
            return "stay"
        return None


def canary(per_seed_items, per_seed_res, layers, position):
    """Per layer: Cohen's d (switch-role items minus stay-role items, per _item_role(., position))
    of the real direction's loading, and the mean of the same d over the random controls."""
    out = {}
    for L in layers:
        real_sw, real_st, rnd_sw, rnd_st = [], [], [], []
        for items, res in zip(per_seed_items, per_seed_res):
            if L not in res:
                continue
            proj = res[L]                                     # [1+R, gen_len]
            for it in items:
                role = _item_role(it, position)
                if role == "switch":
                    real_sw.append(proj[0, it["t"]]); rnd_sw.append(proj[1:, it["t"]])
                elif role == "stay":
                    real_st.append(proj[0, it["t"]]); rnd_st.append(proj[1:, it["t"]])
        d_real = _cohens_d(real_sw, real_st)
        d_rand = None
        if len(rnd_sw) > 1 and len(rnd_st) > 1 and np.asarray(rnd_sw).shape[1] > 0:
            rs, rt = np.asarray(rnd_sw), np.asarray(rnd_st)
            ds = [_cohens_d(rs[:, r], rt[:, r]) for r in range(rs.shape[1])]
            ds = [x for x in ds if x is not None]
            d_rand = float(np.mean(ds)) if ds else None
        out[str(L)] = {"d_real": d_real, "d_random_mean": d_rand,
                       "n_switch": len(real_sw), "n_stay": len(real_st)}
    return out


def main():
    model_path  = sys.argv[1]
    out_dir     = sys.argv[2]
    norms_csv   = sys.argv[3]
    generic_dir = sys.argv[4]
    seed_start  = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    n_seeds     = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    n_random    = int(sys.argv[7]) if len(sys.argv) > 7 else N_RANDOM_DEFAULT
    spaces      = sys.argv[8].split(",") if len(sys.argv) > 8 else ["residual", "mlp"]
    capture_dir = sys.argv[9] if len(sys.argv) > 9 else _default_capture_dir(generic_dir)
    assert set(spaces) <= {"residual", "mlp"}, spaces
    os.makedirs(out_dir, exist_ok=True)

    # load / fit directions before loading the model, so a missing analysis fails fast
    dirs = load_directions(generic_dir, spaces)
    fit_missing_directions(generic_dir, capture_dir, spaces, dirs, norms_csv)
    if not any(dirs[s][p] for s in ("residual", "mlp") for p in POSITIONS):
        raise FileNotFoundError(f"no generic_*_direction_layer*.npy files for spaces {spaces} in {generic_dir} "
                                f"-- has exp5_analysis_single.py finished for this model?")
    model, tok = J.load_model(model_path)
    n_layers = J.get_num_hidden_layers(model)

    pos_layers = {"residual": {}, "mlp": {}}         # pos_layers[space][position] -> sorted [layer,...]
    for space in ("residual", "mlp"):
        for position in POSITIONS:
            pos_layers[space][position] = sorted(L for L in dirs[space][position] if L < n_layers)
            beyond = sorted(set(dirs[space][position]) - set(pos_layers[space][position]))
            if beyond:
                print(f"WARNING: {space}/{position} direction files for layers {beyond} skipped -- "
                      f"model has only {n_layers} layers", flush=True)
    residual_layers = sorted(set(pos_layers["residual"]["pre"]) | set(pos_layers["residual"]["emit"]))
    mlp_layers = sorted(set(pos_layers["mlp"]["pre"]) | set(pos_layers["mlp"]["emit"]))
    if not residual_layers and not mlp_layers:
        raise ValueError(f"every direction file in {generic_dir} is for a layer >= the model's {n_layers} layers")
    print(f"directions from {generic_dir}: residual layers {residual_layers} "
          f"(pre {pos_layers['residual']['pre']}, emit {pos_layers['residual']['emit']}), "
          f"MLP layers {mlp_layers} (pre {pos_layers['mlp']['pre']}, emit {pos_layers['mlp']['emit']}), "
          f"{n_random} random controls per direction", flush=True)

    # dmats[space][L][position]: row 0 = exp5 direction, rows 1..n_random = orthogonal random
    # controls (fixed seed and draw order, so identical in every chunk)
    rng = torch.Generator(device=J.DEVICE).manual_seed(RANDOM_SEED)
    dmats = {"residual": {}, "mlp": {}}
    for space, layers in (("residual", residual_layers), ("mlp", mlp_layers)):
        for L in layers:
            dmats[space][L] = {}
            for position in POSITIONS:
                if L not in dirs[space][position]:
                    continue
                d = torch.nn.functional.normalize(
                    torch.from_numpy(dirs[space][position][L]).to(J.DEVICE, torch.float32), dim=0)
                rows = [d] + [J.random_orthogonal_unit(d, generator=rng) for _ in range(n_random)]
                dmats[space][L][position] = torch.stack(rows)

    norms = N.load_troyer_norms(norms_csv, tok, verbose=False)
    word2cats, all_members = norms["word2cats"], norms["all_members"]
    leading_cache = {}

    msgs = [{"role": "user", "content": PROMPT}]
    gen_in = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(gen_in):
        gen_in = gen_in["input_ids"]   # newer transformers can return a BatchEncoding here
    gen_in = gen_in.to(J.DEVICE)
    attn_mask = torch.ones_like(gen_in)
    prompt_len = gen_in.shape[1]

    kept = []                        # per kept seed: dict of everything to save
    for seed in range(seed_start, seed_start + n_seeds):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(gen_in, attention_mask=attn_mask, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                  do_sample=True, temperature=TEMP, top_p=TOP_P,
                                  pad_token_id=tok.eos_token_id)
        full_ids = out[0]
        gen_len = int(full_ids.shape[0] - prompt_len)
        gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)
        gen_ids = full_ids[prompt_len:].tolist()
        tracked = N.build_tracked_sequence_full(gen_text, gen_ids, prompt_len, all_members, tok,
                                                 cache=leading_cache)
        if len(tracked) < 2:
            print(f"  seed {seed}: {len(tracked)} tracked item(s) -- skipped", flush=True)
            continue
        items, clusters, (cid, tsince, ttosw) = annotate_clusters(tracked, word2cats, prompt_len, gen_len)

        # full forward pass: hidden states for every layer, MLP inputs for MLP-direction layers
        mlp_storage = {L: [] for L in mlp_layers}
        handles = [J.register_mlp_capture_hook(model, L, mlp_storage[L]) for L in mlp_layers]
        try:
            with torch.no_grad():
                out2 = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        rec = {"seed": seed, "gen_len": gen_len, "generated": gen_text, "items": items, "clusters": clusters,
               "cid": cid, "tsince": tsince, "ttosw": ttosw,
               "res": {"pre": {}, "emit": {}}, "resn": {}, "mlp": {"pre": {}, "emit": {}}, "mlpn": {}}
        for L in residual_layers:
            H = out2.hidden_states[L + 1][0].float()[prompt_len:]
            for position, dmat in dmats["residual"][L].items():
                assert H.shape[1] == dmat.shape[1], \
                    f"residual dim {H.shape[1]} != direction dim {dmat.shape[1]} at layer {L} ({position})"
                loadings, resn = project(H, dmat)
                rec["res"][position][L] = loadings
                rec["resn"][L] = resn              # same for both positions
        for L in mlp_layers:
            A = mlp_storage[L][0][0].float()[prompt_len:]
            for position, dmat in dmats["mlp"][L].items():
                assert A.shape[1] == dmat.shape[1], \
                    f"MLP dim {A.shape[1]} != direction dim {dmat.shape[1]} at layer {L} ({position})"
                loadings, mlpn = project(A, dmat)
                rec["mlp"][position][L] = loadings
                rec["mlpn"][L] = mlpn              # same for both positions
        kept.append(rec)
        n_sw = sum(c["ends_in_switch"] for c in clusters)
        print(f"  seed {seed}: {len(items)} items, {len(clusters)} clusters ({n_sw} end in a switch), "
              f"{gen_len} generated tokens", flush=True)

    seed_end = seed_start + n_seeds - 1
    tag = f"seed{seed_start:03d}-{seed_end:03d}"
    n_kept = len(kept)
    T = max((r["gen_len"] for r in kept), default=0)

    def pad2(rows, fill, dtype):                       # list of [T_i] -> [n_kept, T]
        a = np.full((n_kept, T), fill, dtype)
        for i, r in enumerate(rows):
            a[i, :len(r)] = r
        return a

    def pad3(rows, fill):                              # list of [1+R, T_i] -> [n_kept, 1+R, T]
        a = np.full((n_kept, 1 + n_random, T), fill, np.float32)
        for i, r in enumerate(rows):
            a[i, :, :r.shape[1]] = r
        return a

    arrays = {"seeds": np.array([r["seed"] for r in kept], np.int64),
              "prompt_len": np.full(n_kept, prompt_len, np.int64),
              "gen_len": np.array([r["gen_len"] for r in kept], np.int64),
              "cluster_id": pad2([r["cid"] for r in kept], -1, np.int32),
              "tok_since_start": pad2([r["tsince"] for r in kept], -1, np.int32),
              "tok_to_switch": pad2([r["ttosw"] for r in kept], -1, np.int32)}
    for L in residual_layers:
        for position in POSITIONS:
            if position not in dmats["residual"][L]:
                continue
            arrays[f"res_L{L:02d}_{position}"] = pad3([r["res"][position][L] for r in kept], np.nan)
        arrays[f"resnorm_L{L:02d}"] = pad2([r["resn"][L] for r in kept], np.nan, np.float32)
    for L in mlp_layers:
        for position in POSITIONS:
            if position not in dmats["mlp"][L]:
                continue
            arrays[f"mlp_L{L:02d}_{position}"] = pad3([r["mlp"][position][L] for r in kept], np.nan)
        arrays[f"mlpnorm_L{L:02d}"] = pad2([r["mlpn"][L] for r in kept], np.nan, np.float32)
    np.savez_compressed(os.path.join(out_dir, f"exp9_{tag}.npz"), **arrays)

    can = {}
    for position in POSITIONS:
        layers_here = [L for L in residual_layers if position in dmats["residual"][L]]
        if not layers_here:
            continue
        can[position] = canary([r["items"] for r in kept],
                                [r["res"][position] for r in kept],
                                layers_here, position)

    meta = {"seed_start": seed_start, "n_seeds": n_seeds, "seeds_kept": [r["seed"] for r in kept],
            "n_random": n_random, "random_seed": RANDOM_SEED, "generic_dir": generic_dir, "spaces": spaces,
            "residual_layers": residual_layers, "mlp_layers": mlp_layers,
            "residual_layers_pre": pos_layers["residual"]["pre"], "residual_layers_emit": pos_layers["residual"]["emit"],
            "mlp_layers_pre": pos_layers["mlp"]["pre"], "mlp_layers_emit": pos_layers["mlp"]["emit"],
            "prompt_len": prompt_len, "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
            "per_seed": [{"seed": r["seed"], "prompt_len": prompt_len, "gen_len": r["gen_len"],
                          "generated": r["generated"], "items": r["items"], "clusters": r["clusters"]}
                         for r in kept],
            "canary": can}
    with open(os.path.join(out_dir, f"exp9_{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    n_clusters = sum(len(r["clusters"]) for r in kept)
    n_switch_clusters = sum(c["ends_in_switch"] for r in kept for c in r["clusters"])
    print(json.dumps({"seed_start": seed_start, "n_seeds": n_seeds, "seeds_kept": n_kept,
                      "n_clusters": n_clusters, "n_clusters_ending_in_switch": n_switch_clusters}, indent=2))
    print("canary (Cohen's d, switch-role minus stay-role items per _item_role; real direction vs random controls):")
    for position, layer_results in can.items():
        print(f"  -- {position} --")
        for L in residual_layers:
            if str(L) not in layer_results:
                continue
            c = layer_results[str(L)]
            fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"
            print(f"  layer {L:2d}: real {fmt(c['d_real'])}  random {fmt(c['d_random_mean'])}  "
                  f"(n_switch={c['n_switch']}, n_stay={c['n_stay']})", flush=True)


if __name__ == "__main__":
    main()
