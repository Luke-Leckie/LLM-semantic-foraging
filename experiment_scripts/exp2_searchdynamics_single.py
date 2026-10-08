# exp2_searchdynamics_single.py
# Usage: python3 exp2_searchdynamics_single.py <layer> <model_path> <out_dir> <norms_csv> <seed_start> <n_seeds> [top_k]
#
# Experiment 2: do patch switches during animal fluency coincide with depletion
# of the J-space patch supply? For every adjacent pair of emitted animals
# (w -> w_next), at w's position:
#   * switch  = w and w_next share no category (Troyer norms, via jlens_norms)
#   * supply  = # not-yet-emitted, single-token members of w's category that are
#               in the top-K J-space (within_z_activation: their mean z-activation)
#   * dest_supply / dest_z_activation: the same for w_next's category
#   * emitted_supply / emitted_z_activation: the same for w's already-emitted members
#   * patch_pool_size / total_category_size: remaining / total category size
#     (symbolic controls with no J-space restriction)
#   * surprisal of w_next (full vocab and renormalised over animal probes)
# Each seed is one sampled rollout with its own summary record plus raw events;
# statistics across seeds are computed downstream.
#
# top_k (default 25) sets the J-space occupancy budget for the *_supply measures.
# If env var JLENS_PT points to a jacobian-lens checkpoint, activations are read
# out through that lens; otherwise J-lens vectors are estimated from a corpus.
#
# Output: <out_dir>/exp2_layer<LL>_seed<start>-<end>.json (+ probe vectors as .npy)
import sys, os, math, json, numpy as np, torch
import jlens_common as J
import jlens_norms as N

TOP_K     = 25       # default J-space occupancy budget (overridable via CLI)
TEMP      = 0.9
TOP_P     = 0.95
LN2       = math.log(2)


def surprisal_bits_full(logit_row, token_id):
    """-log2 P(token_id | context) over the full vocab."""
    lp = torch.log_softmax(logit_row, dim=-1)[token_id]
    return float(-lp / LN2)


def surprisal_bits_subset(logit_row, subset_ids, idx_in_subset):
    """-log2 P(token | context) renormalized over `subset_ids` (the animal probes)."""
    lp = torch.log_softmax(logit_row[subset_ids], dim=-1)[idx_in_subset]
    return float(-lp / LN2)


def run_one_seed(model, tok, gen_in, attn_mask, layer, jlens_pt, Jmat, V,
                  cand_ids, cand_ids_t, probe_words, cidx, word_of_id,
                  word2cats, patch_pool, seed, top_k=TOP_K):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(gen_in, attention_mask=attn_mask, max_new_tokens=220,
                              do_sample=True, temperature=TEMP, top_p=TOP_P,
                              pad_token_id=tok.eos_token_id)
    full_ids = out[0]
    prompt_len = gen_in.shape[1]
    gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)

    if jlens_pt:
        H_raw, logits = J.read_activations(model, tok, full_ids, layer,
                                            return_logits=True, normalize=False)
        A = J.lens_forward_logits(model, H_raw, layer, Jmat, cand_ids).cpu().numpy()
    else:
        H, logits = J.read_activations(model, tok, full_ids, layer,
                                        return_logits=True)        # [S, D], [S, vocab]
        A = (H @ V.T).cpu().numpy()                                # [S, C]

    # z-score each probe over the generated region
    gen_slice = slice(prompt_len, A.shape[0])
    Az = (A - A[gen_slice].mean(0, keepdims=True)) / (A[gen_slice].std(0, keepdims=True) + 1e-6)

    gen_ids = full_ids[prompt_len:].tolist()
    # tracked items carry a gap flag: True when an unrecognised item (e.g. a
    # multi-word animal) sits between two probe emissions, so they are not adjacent
    tracked = N.build_tracked_sequence(gen_text, gen_ids, prompt_len, word_of_id)
    seq = [(pos, w) for pos, w, _ in tracked]

    events = []
    emitted = set()
    run_len = 0
    for n in range(len(seq) - 1):
        e_abs, w = seq[n]
        next_abs, w_next = seq[n + 1]
        emitted.add(w)
        run_len += 1
        if tracked[n + 1][2]:
            run_len = 0
            continue
        src = e_abs
        if src >= A.shape[0]:
            continue
        pool = [k for k in patch_pool[w] if probe_words[k] not in emitted]
        if not pool:
            continue
        topk = set(np.argpartition(A[src], -top_k)[-top_k:])
        supply = int(sum(1 for k in pool if k in topk))
        within_z = float(np.mean([Az[src, k] for k in pool]))
        switch = N.is_switch(w, w_next, word2cats)

        # already-emitted members of w's category (complement of `pool`)
        emitted_pool = [k for k in patch_pool[w] if probe_words[k] in emitted]
        emitted_supply = int(sum(1 for k in emitted_pool if k in topk))
        emitted_z = float(np.mean([Az[src, k] for k in emitted_pool])) if emitted_pool else None

        # all single-token members of w's category, emitted or not (no J-space restriction)
        total_category_size = len(patch_pool[w])

        pred_pos = next_abs - 1
        next_tok_id = cand_ids[cidx[w_next]]
        surp_full = surprisal_bits_full(logits[pred_pos], next_tok_id)
        surp_cand = surprisal_bits_subset(logits[pred_pos], cand_ids_t, cidx[w_next])

        # destination category: is w_next's category already active at `src`?
        dest_pool = [k for k in patch_pool[w_next] if probe_words[k] not in emitted]
        dest_supply = int(sum(1 for k in dest_pool if k in topk)) if dest_pool else None
        dest_z = float(np.mean([Az[src, k] for k in dest_pool])) if dest_pool else None

        events.append({
            "pos_in_list": n, "word": w, "next": w_next,
            "run_len": run_len, "patch_pool_size": len(pool),
            "supply": supply, "within_z_activation": within_z, "switch": bool(switch),
            "surprisal_full_bits": surp_full,
            "surprisal_candset_bits": surp_cand,
            "cats": sorted(word2cats.get(w, [])),
            "dest_pool_size": len(dest_pool),
            "dest_supply": dest_supply,
            "dest_z_activation": dest_z,
            "emitted_pool_size": len(emitted_pool),
            "emitted_supply": emitted_supply,
            "emitted_z_activation": emitted_z,
            "total_category_size": total_category_size,
        })
        if switch:
            run_len = 0

    sw = np.array([e["supply"] for e in events if e["switch"]])
    st = np.array([e["supply"] for e in events if not e["switch"]])
    sfw = np.array([e["surprisal_full_bits"] for e in events if e["switch"]])
    sfs = np.array([e["surprisal_full_bits"] for e in events if not e["switch"]])
    scw = np.array([e["surprisal_candset_bits"] for e in events if e["switch"]])
    scs = np.array([e["surprisal_candset_bits"] for e in events if not e["switch"]])
    dsw = np.array([e["dest_supply"] for e in events if e["switch"] and e["dest_supply"] is not None])
    dst = np.array([e["dest_supply"] for e in events if not e["switch"] and e["dest_supply"] is not None])
    dzw = np.array([e["dest_z_activation"] for e in events if e["switch"] and e["dest_z_activation"] is not None])
    dzt = np.array([e["dest_z_activation"] for e in events if not e["switch"] and e["dest_z_activation"] is not None])
    wzw = np.array([e["within_z_activation"] for e in events if e["switch"]])
    wzt = np.array([e["within_z_activation"] for e in events if not e["switch"]])
    esw = np.array([e["emitted_supply"] for e in events if e["switch"]])
    est = np.array([e["emitted_supply"] for e in events if not e["switch"]])
    ezw = np.array([e["emitted_z_activation"] for e in events
                     if e["switch"] and e["emitted_z_activation"] is not None])
    ezt = np.array([e["emitted_z_activation"] for e in events
                     if not e["switch"] and e["emitted_z_activation"] is not None])
    tcw = np.array([e["total_category_size"] for e in events if e["switch"]])
    tct = np.array([e["total_category_size"] for e in events if not e["switch"]])

    return {
        "seed": seed,
        "generated": gen_text,
        "n_switch_events": int(sw.size), "n_stay_events": int(st.size),
        "mean_supply_before_switch": float(sw.mean()) if sw.size else None,
        "mean_supply_before_stay":   float(st.mean()) if st.size else None,
        "mean_surprisal_full_bits_switch": float(sfw.mean()) if sfw.size else None,
        "mean_surprisal_full_bits_stay":   float(sfs.mean()) if sfs.size else None,
        "mean_surprisal_candset_bits_switch": float(scw.mean()) if scw.size else None,
        "mean_surprisal_candset_bits_stay":   float(scs.mean()) if scs.size else None,
        "mean_dest_supply_switch": float(dsw.mean()) if dsw.size else None,
        "mean_dest_supply_stay":   float(dst.mean()) if dst.size else None,
        "mean_dest_z_activation_switch": float(dzw.mean()) if dzw.size else None,
        "mean_dest_z_activation_stay":   float(dzt.mean()) if dzt.size else None,
        "mean_within_z_activation_switch": float(wzw.mean()) if wzw.size else None,
        "mean_within_z_activation_stay":   float(wzt.mean()) if wzt.size else None,
        "mean_emitted_supply_switch": float(esw.mean()) if esw.size else None,
        "mean_emitted_supply_stay":   float(est.mean()) if est.size else None,
        "mean_total_category_size_switch": float(tcw.mean()) if tcw.size else None,
        "mean_total_category_size_stay":   float(tct.mean()) if tct.size else None,
        "mean_emitted_z_activation_switch": float(ezw.mean()) if ezw.size else None,
        "mean_emitted_z_activation_stay":   float(ezt.mean()) if ezt.size else None,
        "events": events,
    }


def main():
    layer      = int(sys.argv[1])
    model_path = sys.argv[2]
    out_dir    = sys.argv[3]
    norms_csv  = sys.argv[4]
    seed_start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    n_seeds    = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    top_k      = int(sys.argv[7]) if len(sys.argv) > 7 else TOP_K
    os.makedirs(out_dir, exist_ok=True)

    model, tok = J.load_model(model_path)

    norms = N.load_troyer_norms(norms_csv, tok)
    word2cats  = norms["word2cats"]
    categories = norms["categories"]
    probe_ids  = norms["probe_ids"]                 # {word: token_id}, single-token
    probe_words = list(probe_ids.keys())
    cand_ids = [probe_ids[w] for w in probe_words]
    cidx = {w: k for k, w in enumerate(probe_words)}
    word_of_id = N.build_word_of_id(tok, probe_ids)
    print(f"layer {layer}: {len(probe_words)} single-token probes, top_k={top_k}, "
          f"seeds {seed_start}-{seed_start + n_seeds - 1}", flush=True)

    patch_pool = {}
    for w in probe_words:
        nbrs = N.within_patch_members(w, word2cats, categories)
        patch_pool[w] = [cidx[m] for m in nbrs if m in cidx]

    cand_ids_t = torch.as_tensor(cand_ids, device=J.DEVICE)
    jlens_pt = os.environ.get("JLENS_PT")
    seed_end = seed_start + n_seeds - 1
    layer_tag = f"{layer:02d}_seed{seed_start:03d}-{seed_end:03d}"

    V, Jmat = None, None
    if jlens_pt:
        Jmat = J.load_jlens_pt(jlens_pt)["J"]
        W_U, _ = J.get_unembed_and_norm(model)
        V_lens = torch.nn.functional.normalize(
            W_U[cand_ids_t].float() @ Jmat[layer].to(J.DEVICE, torch.float32), dim=1)
        np.save(os.path.join(out_dir, f"jlens_V_layer{layer_tag}.npy"), V_lens.cpu().numpy())
    else:
        corpus = J.load_corpus(120, os.environ.get("JLENS_CORPUS"))
        V = J.compute_jlens_vectors(model, tok, cand_ids, layer, corpus)   # [C, D]
        np.save(os.path.join(out_dir, f"jlens_V_layer{layer_tag}.npy"), V.cpu().numpy())

    prompt = ("Name as many different animals as you can, one after another, "
              "separated by commas. Just the list.")
    msgs = [{"role": "user", "content": prompt}]
    gen_in = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(gen_in):
        gen_in = gen_in["input_ids"]   # newer transformers can return a BatchEncoding here
    gen_in = gen_in.to(J.DEVICE)
    attn_mask = torch.ones_like(gen_in)

    records = []
    for seed in range(seed_start, seed_start + n_seeds):
        print(f"  seed {seed} ...", flush=True)
        records.append(run_one_seed(model, tok, gen_in, attn_mask, layer, jlens_pt, Jmat, V,
                                     cand_ids, cand_ids_t, probe_words, cidx, word_of_id,
                                     word2cats, patch_pool, seed, top_k=top_k))

    out = {
        "layer": layer, "top_k": top_k,
        "probe_words": probe_words,          # row order matches jlens_V_layer*.npy
        "used_precomputed_lens": bool(jlens_pt),
        "jlens_pt": jlens_pt,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "records": records,
    }
    with open(os.path.join(out_dir, f"exp2_layer{layer_tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    sw = [r["mean_supply_before_switch"] for r in records if r["mean_supply_before_switch"] is not None]
    st = [r["mean_supply_before_stay"] for r in records if r["mean_supply_before_stay"] is not None]
    print(json.dumps({
        "layer": layer, "n_seeds": n_seeds,
        "mean_of_per_seed_mean_supply_switch": float(np.mean(sw)) if sw else None,
        "mean_of_per_seed_mean_supply_stay": float(np.mean(st)) if st else None,
    }, indent=2))


if __name__ == "__main__":
    main()
