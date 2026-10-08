# exp5_switch_neurons_full_single.py
# Usage: python3 exp5_switch_neurons_full_single.py <model_path> <out_dir> <norms_csv> <seed_start> <n_seeds>
#
# Experiment 5 (capture): records residual-stream and MLP activations at every
# adjacent switch/stay transition during sampled animal-fluency rollouts, for
# analysis by exp5_analysis_single.py.
#   * Every recognised animal, single- or multi-token, is a tracked item (a
#     multi-token word is located at its leading token).
#   * Positions: "pre" = the current item (before the next item is chosen),
#     "emit" = the next item.
#   * Residual: every even layer 2..n_layers-2. MLP (down_proj input): 5 of
#     those layers, evenly spaced.
#   * Transitions across an unrecognised item, or from a word whose category
#     has no other members, are skipped.
#
# Output: <out_dir>/exp5_seed<start>-<end>.npz (activations, one row per event)
#         <out_dir>/exp5_seed<start>-<end>_meta.json (event metadata, same row order)
import sys, os, json, numpy as np, torch
import jlens_common as J
import jlens_norms as N

TEMP, TOP_P = 0.9, 0.95
GEN_MAX_NEW_TOKENS = 220


def pick_layers(n_layers):
    """Even layers 2..n_layers-2 for residual capture; 5 of them, evenly spaced,
    for MLP capture."""
    residual_layers = list(range(2, n_layers, 2))
    if len(residual_layers) <= 5:
        mlp_layers = list(residual_layers)
    else:
        idx = np.linspace(0, len(residual_layers) - 1, 5).round().astype(int)
        mlp_layers = sorted(set(residual_layers[i] for i in idx))
    return residual_layers, mlp_layers


def main():
    model_path = sys.argv[1]
    out_dir    = sys.argv[2]
    norms_csv  = sys.argv[3]
    seed_start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    n_seeds    = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    os.makedirs(out_dir, exist_ok=True)

    model, tok = J.load_model(model_path)
    residual_layers, mlp_layers = pick_layers(J.get_num_hidden_layers(model))
    print(f"residual layers: {residual_layers}", flush=True)
    print(f"MLP layers: {mlp_layers}", flush=True)

    norms = N.load_troyer_norms(norms_csv, tok)
    word2cats   = norms["word2cats"]
    categories  = norms["categories"]
    probe_ids   = norms["probe_ids"]
    all_members = norms["all_members"]
    print(f"{len(all_members)} recognized animals ({len(probe_ids)} single-token), "
          f"seeds {seed_start}-{seed_start + n_seeds - 1}", flush=True)
    leading_cache = {}   # memo for word detection, shared across seeds

    prompt = ("Name as many different animals as you can, one after another, "
              "separated by commas. Just the list.")
    msgs = [{"role": "user", "content": prompt}]
    gen_in = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(gen_in):
        gen_in = gen_in["input_ids"]   # newer transformers can return a BatchEncoding here
    gen_in = gen_in.to(J.DEVICE)
    attn_mask = torch.ones_like(gen_in)

    events_meta = []
    residual_pre  = {L: [] for L in residual_layers}
    residual_emit = {L: [] for L in residual_layers}
    mlp_pre  = {L: [] for L in mlp_layers}
    mlp_emit = {L: [] for L in mlp_layers}

    for seed in range(seed_start, seed_start + n_seeds):
        print(f"  seed {seed} ...", flush=True)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(gen_in, attention_mask=attn_mask, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                  do_sample=True, temperature=TEMP, top_p=TOP_P,
                                  pad_token_id=tok.eos_token_id)
        full_ids = out[0]
        prompt_len = gen_in.shape[1]
        gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)
        gen_ids = full_ids[prompt_len:].tolist()

        tracked = N.build_tracked_sequence_full(gen_text, gen_ids, prompt_len, all_members,
                                                 tok, cache=leading_cache)
        seq = [(pos, w) for pos, w, _ in tracked]
        if len(seq) < 2:
            continue

        # full forward pass over the generated sequence to capture hidden states and
        # MLP activations at every position
        mlp_storage = {L: [] for L in mlp_layers}
        handles = [J.register_mlp_capture_hook(model, L, mlp_storage[L]) for L in mlp_layers]
        try:
            with torch.no_grad():
                out2 = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        hs = {L: out2.hidden_states[L + 1][0].float().cpu().numpy() for L in residual_layers}
        mlp_acts = {L: mlp_storage[L][0][0].float().cpu().numpy() for L in mlp_layers}  # [S, intermediate]
        seq_len = hs[residual_layers[0]].shape[0]

        for n in range(len(seq) - 1):
            e_abs, w = seq[n]
            next_abs, w_next = seq[n + 1]
            if tracked[n + 1][2]:
                continue   # not adjacent (unrecognised item in between)
            src = e_abs
            if src >= seq_len or next_abs >= seq_len:
                continue
            if not N.within_patch_members(w, word2cats, categories):
                continue   # category has no other members
            switch = N.is_switch(w, w_next, word2cats)
            src_cat = sorted(word2cats.get(w, ["?"]))[0]       # primary (alphabetically first) category
            dest_cat = sorted(word2cats.get(w_next, ["?"]))[0]

            events_meta.append({
                "seed": seed, "pos_in_list": n, "switch": bool(switch),
                "word": w, "next": w_next, "src_cat": src_cat, "dest_cat": dest_cat,
                "pre_pos": int(src), "emit_pos": int(next_abs),
                "word_is_multitoken": w not in probe_ids, "next_is_multitoken": w_next not in probe_ids,
            })
            for L in residual_layers:
                residual_pre[L].append(hs[L][src])
                residual_emit[L].append(hs[L][next_abs])
            for L in mlp_layers:
                mlp_pre[L].append(mlp_acts[L][src])
                mlp_emit[L].append(mlp_acts[L][next_abs])

    seed_end = seed_start + n_seeds - 1
    tag = f"seed{seed_start:03d}-{seed_end:03d}"

    npz_arrays = {}
    for L in residual_layers:
        npz_arrays[f"residual_pre_layer{L:02d}"] = np.stack(residual_pre[L]) if residual_pre[L] else np.zeros((0,))
        npz_arrays[f"residual_emit_layer{L:02d}"] = np.stack(residual_emit[L]) if residual_emit[L] else np.zeros((0,))
    for L in mlp_layers:
        npz_arrays[f"mlp_pre_layer{L:02d}"] = np.stack(mlp_pre[L]) if mlp_pre[L] else np.zeros((0,))
        npz_arrays[f"mlp_emit_layer{L:02d}"] = np.stack(mlp_emit[L]) if mlp_emit[L] else np.zeros((0,))
    np.savez_compressed(os.path.join(out_dir, f"exp5_{tag}.npz"), **npz_arrays)

    out = {
        "seed_start": seed_start, "n_seeds": n_seeds,
        "residual_layers": residual_layers, "mlp_layers": mlp_layers,
        "n_events": len(events_meta),
        "events": events_meta,
    }
    with open(os.path.join(out_dir, f"exp5_{tag}_meta.json"), "w") as f:
        json.dump(out, f, indent=2)

    n_switch = sum(e["switch"] for e in events_meta)
    n_multi = sum(e["word_is_multitoken"] or e["next_is_multitoken"] for e in events_meta)
    print(json.dumps({
        "seed_start": seed_start, "n_seeds": n_seeds,
        "n_events": len(events_meta), "n_switch": n_switch, "n_stay": len(events_meta) - n_switch,
        "n_events_involving_multitoken_word": n_multi,
    }, indent=2))


if __name__ == "__main__":
    main()
