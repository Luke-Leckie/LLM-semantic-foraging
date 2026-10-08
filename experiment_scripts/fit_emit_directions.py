# fit_emit_directions.py
# Usage: python3 fit_emit_directions.py <norms_csv> [results_exp5_full_dir:results_exp5_full_analysis_dir ...]
#   (with no directory args, fits for all 5 model pairs this repo already has locally)
#
# One-off LOCAL companion to exp5_analysis_single.py: fits and saves the "emit"-position
# sibling of generic_switch_direction_layer<NN>.npy / generic_mlp_direction_layer<NN>.npy --
# exp5_analysis_single.py's fit_generic_direction is only ever called for position=="pre" (see
# that script's header for why: "pre" is the causally-relevant, anticipatory position for
# steering). This produces the "emit" (destination item's own token, i.e. AT the switch, not
# anticipating it) direction too, purely for the OBSERVATIONAL comparison in exp9 -- does the
# ramp-while-clustering effect look different projected onto "what commit-to-switch looks like
# at the moment it happens" versus "what predicts it beforehand"?
#
# Runs ENTIRELY LOCALLY, no cluster/GPU needed: fit_generic_direction is a single PCA + logistic
# regression fit (no permutation sweep, no held-out CV -- that's held_out_probe_pca, which this
# script does NOT call) on already-captured activation arrays. At Gemma-9B's ~3400 events x
# 3584-dim, this is seconds per layer, not the "too heavy for a laptop" cost that moved the full
# exp5_analysis sweep to the cluster in the first place.
#
# READS: <results_exp5_full_dir>/exp5_seed*[_meta].json -- the RAW captured activations
#        (exp5_switch_neurons_full_single.py's output), which already contain BOTH
#        residual_emit_layer<NN> / mlp_emit_layer<NN> arrays alongside the _pre ones used by
#        exp5_analysis_single.py -- so no new capture run is needed either.
# WRITES: <results_exp5_full_analysis_dir>/generic_switch_direction_emit_layer<NN>.npy (residual)
#         and generic_mlp_direction_emit_layer<NN>.npy (MLP) -- alongside, not replacing, the
#         existing "pre" files exp5_analysis_single.py already wrote there.
#
# Uses the SAME multi-label category counting as the (already fixed) exp5_analysis_single.py --
# imported directly, not reimplemented, so this can't silently drift out of sync with that fix.
import sys, os, glob, json
import numpy as np
import jlens_norms as N
import pandas as pd
from exp5_analysis_single import fit_generic_direction

DEFAULT_PAIRS = [
    ("/media/ll16598/432C-7284/jlens/results_exp5_full", "/media/ll16598/432C-7284/jlens/results_exp5_full_analysis"),
    ("/media/ll16598/432C-7284/jlens/results_exp5_full_2b", "/media/ll16598/432C-7284/jlens/results_exp5_full_analysis_2b"),
    ("/media/ll16598/432C-7284/jlens/results_exp5_full_qwen", "/media/ll16598/432C-7284/jlens/results_exp5_full_analysis_qwen"),
    ("/media/ll16598/432C-7284/jlens/results_exp5_full_llama31", "/media/ll16598/432C-7284/jlens/results_exp5_full_analysis_llama31"),
    ("/media/ll16598/432C-7284/jlens/results_exp5_full_gemma3", "/media/ll16598/432C-7284/jlens/results_exp5_full_analysis_gemma3"),
]
DEFAULT_NORMS_CSV = "extenden_nundin_norms.csv"


def load_exp5_events(d):
    """Just the events DataFrame (cheap -- it's the small JSON metadata, not the
    activation arrays), in seed_start order. Split out of exp5_analysis_single.load_exp5
    so the caller can load ONE layer's array at a time afterward instead of every
    layer/position/space at once -- load_exp5's all-at-once concatenation OOM-killed this
    script on this machine's free RAM (residual: 20 layers x 2 positions x ~3400x3584
    float32 = ~2GB; MLP: 5 layers x 2 positions x ~3400x14336 float32 = ~2GB;
    load_exp5's own list-then-concatenate pattern transiently holds ~2x that -- comfortably
    OOMs at the ~6GB this machine had free)."""
    meta_fps = sorted(glob.glob(os.path.join(d, "exp5_seed*_meta.json")),
                       key=lambda fp: json.load(open(fp))["seed_start"])
    all_events = []
    for meta_fp in meta_fps:
        all_events.extend(json.load(open(meta_fp))["events"])
    return pd.DataFrame(all_events), meta_fps


def load_one_layer_array(meta_fps, key):
    """The single npz array `key` (e.g. "residual_emit_layer20"), concatenated across
    chunks in the SAME seed_start order load_exp5_events used -- so its rows stay aligned
    with that DataFrame's rows. Returns None if no chunk has this key. Only ONE layer's
    worth of data is ever in memory at a time this way."""
    parts = []
    for meta_fp in meta_fps:
        npz_fp = meta_fp.replace("_meta.json", ".npz")
        with np.load(npz_fp) as npz:
            if key not in npz.files:
                return None
            parts.append(npz[key])
    return np.concatenate(parts, axis=0)


def layers_captured(meta_fps, space, position):
    """Layer numbers with a non-empty `{space}_{position}_layer<NN>` array in the FIRST
    chunk file (every chunk captures the same layer set, per exp5_switch_neurons_full_single.py
    -- checking one file is enough and avoids opening every chunk twice)."""
    if not meta_fps:
        return []
    prefix = f"{space}_{position}_layer"
    with np.load(meta_fps[0].replace("_meta.json", ".npz")) as npz:
        keys = npz.files
    return sorted(int(k[len(prefix):]) for k in keys if k.startswith(prefix))


def main():
    norms_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NORMS_CSV
    pairs = DEFAULT_PAIRS
    if len(sys.argv) > 2:
        pairs = [tuple(a.split(":", 1)) for a in sys.argv[2:]]

    word2cats = N.load_troyer_norms(norms_csv, N.WhitespaceTokStub(), verbose=False)["word2cats"]

    for results_dir, out_dir in pairs:
        if not os.path.isdir(results_dir):
            print(f"SKIP: {results_dir} not found", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {results_dir} -> {out_dir} ===", flush=True)
        events_df, meta_fps = load_exp5_events(results_dir)
        print(f"  {len(events_df)} events loaded from {len(meta_fps)} chunks (events only -- arrays loaded "
              f"per layer below)", flush=True)

        for space, tag in (("residual", "generic_switch_direction"), ("mlp", "generic_mlp_direction")):
            layers = layers_captured(meta_fps, space, "emit")
            for L in layers:
                out_fp = os.path.join(out_dir, f"{tag}_emit_layer{L:02d}.npy")
                if os.path.exists(out_fp):
                    print(f"  {space} layer {L:2d}: {out_fp} already exists, skipping", flush=True)
                    continue
                arr = load_one_layer_array(meta_fps, f"{space}_emit_layer{L:02d}")
                direction = fit_generic_direction(events_df, arr, word2cats, n_pca=30)
                del arr
                if direction is None:
                    print(f"  {space} layer {L:2d}: fit_generic_direction returned None (too few valid events), skipped", flush=True)
                    continue
                np.save(out_fp, direction)
                print(f"  {space} layer {L:2d}: saved {os.path.basename(out_fp)} (dim={direction.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
