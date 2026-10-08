# exp2_ext_supply_single.py
# Usage: python3 exp2_ext_supply_single.py <layer> <model_path> <out_dir> <norms_csv> <seed_start> <n_seeds> [top_k]
#
# Tokenization-robust version of exp2's supply measure. J-lens directions exist
# only for single-token candidates, so exp2's `supply` ignores multi-token
# category members. Here each multi-token member is probed via its leading token,
# and two event streams are computed from the same rollouts:
#   * events_orig: single-token items only, transitions across an unrecognised
#     item skipped (identical to exp2); field `supply`.
#   * events_ext:  every recognised single- or multi-token animal is a tracked
#     item; field `supply_ext` (single- plus multi-token members in the top-K).
#
# Limitation: multi-token words sharing a leading token cannot be told apart, so
# one active leading token can count several words; treat supply_ext as an
# upper-bound sensitivity check rather than a replacement for supply.
#
# If env var JLENS_PT points to a jacobian-lens checkpoint, activations are read
# out through that lens; otherwise J-lens vectors are estimated from a corpus.
#
# Output: <out_dir>/exp2ext_layer<LL>_seed<start>-<end>.json (+ probe vectors as .npy)
import sys, os, math, json, numpy as np, torch
import jlens_common as J
import jlens_norms as N

TOP_K     = 25
TEMP      = 0.9
TOP_P     = 0.95
LN2       = math.log(2)


def surprisal_bits_full(logit_row, token_id):
    lp = torch.log_softmax(logit_row, dim=-1)[token_id]
    return float(-lp / LN2)


def surprisal_bits_subset(logit_row, subset_ids, idx_in_subset):
    lp = torch.log_softmax(logit_row[subset_ids], dim=-1)[idx_in_subset]
    return float(-lp / LN2)


def run_one_seed(model, tok, gen_in, attn_mask, layer, jlens_pt, Jmat, V,
                  cand_ids_ext, cand_ids_ext_t, C, probe_words, cidx, word_of_id,
                  word2cats, categories, all_members, patch_pool, pool_ext, leading_cache,
                  seed, top_k=TOP_K):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(gen_in, attention_mask=attn_mask, max_new_tokens=220,
                              do_sample=True, temperature=TEMP, top_p=TOP_P,
                              pad_token_id=tok.eos_token_id)
    full_ids = out[0]
    prompt_len = gen_in.shape[1]
    gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)

    # A_ext columns: [0:C) single-token candidates (as in exp2), [C:) leading-token
    # groups for multi-token members. `supply` uses only the first C columns.
    if jlens_pt:
        H_raw, logits = J.read_activations(model, tok, full_ids, layer,
                                            return_logits=True, normalize=False)
        A_ext = J.lens_forward_logits(model, H_raw, layer, Jmat, cand_ids_ext).cpu().numpy()
    else:
        H, logits = J.read_activations(model, tok, full_ids, layer, return_logits=True)
        A_ext = (H @ V.T).cpu().numpy()

    gen_ids = full_ids[prompt_len:].tolist()

    # --- "orig" event stream: single-token items only (as in exp2) ---
    tracked_orig = N.build_tracked_sequence(gen_text, gen_ids, prompt_len, word_of_id)
    events_orig = []
    run_len = 0
    emitted = set()
    for n in range(len(tracked_orig) - 1):
        e_abs, w, _ = tracked_orig[n]
        next_abs, w_next, gap_next = tracked_orig[n + 1]
        emitted.add(w)
        run_len += 1
        if gap_next:
            run_len = 0
            continue
        src = e_abs
        if src >= A_ext.shape[0]:
            continue
        pool = [k for k in patch_pool[w] if probe_words[k] not in emitted]
        if not pool:
            continue
        topk_orig = set(np.argpartition(A_ext[src, :C], -top_k)[-top_k:])
        supply = int(sum(1 for k in pool if k in topk_orig))
        switch = N.is_switch(w, w_next, word2cats)
        events_orig.append({"pos_in_list": n, "word": w, "next": w_next,
                             "run_len": run_len, "switch": bool(switch), "supply": supply})
        if switch:
            run_len = 0

    # --- "ext" event stream: single- and multi-token tracked items ---
    tracked_ext = N.build_tracked_sequence_full(gen_text, gen_ids, prompt_len, all_members,
                                                 tok, cache=leading_cache)
    events_ext = []
    run_len = 0
    for n in range(len(tracked_ext) - 1):
        e_abs, w, _ = tracked_ext[n]
        next_abs, w_next, gap_next = tracked_ext[n + 1]
        run_len += 1
        if gap_next:
            run_len = 0
            continue
        src = e_abs
        if src >= A_ext.shape[0]:
            continue
        full_emitted_before = frozenset(it for _, it, _ in tracked_ext[:n + 1])

        pool_single_ext = [k for k in pool_ext[w]["single"] if probe_words[k] not in full_emitted_before]
        pool_multi_ext = [(g, m) for g, m in pool_ext[w]["multi"] if m not in full_emitted_before]
        topk_ext = set(np.argpartition(A_ext[src], -top_k)[-top_k:])
        supply_single = int(sum(1 for k in pool_single_ext if k in topk_ext))
        # count words, not columns (a word can have lowercase and Title-Case leading ids)
        multi_words_active = {m for g, m in pool_multi_ext if (C + g) in topk_ext}
        supply_multi = len(multi_words_active)
        supply_ext = supply_single + supply_multi

        switch = N.is_switch(w, w_next, word2cats)
        total_category_size_true = len(N.within_patch_members(w, word2cats, categories))

        events_ext.append({
            "pos_in_list": n, "word": w, "next": w_next,
            "run_len": run_len, "switch": bool(switch),
            "patch_pool_size_ext": len(pool_single_ext) + len(pool_multi_ext), "supply_ext": supply_ext,
            "supply_ext_single_part": supply_single, "supply_ext_multi_part": supply_multi,
            "total_category_size_true": total_category_size_true,
            "word_is_multitoken": w not in cidx, "next_is_multitoken": w_next not in cidx,
            "cats": sorted(word2cats.get(w, [])),
        })
        if switch:
            run_len = 0

    # surprisal of the next item (single-token `next` only)
    def add_surprisal(events, tracked):
        by_n = {e["pos_in_list"]: e for e in events}
        for n, e in by_n.items():
            next_abs = tracked[n + 1][0]
            w_next = tracked[n + 1][1]
            pred_pos = next_abs - 1
            if w_next not in cidx:
                continue
            next_tok_id = cand_ids_ext[cidx[w_next]]
            e["surprisal_full_bits"] = surprisal_bits_full(logits[pred_pos], next_tok_id)
            e["surprisal_candset_bits"] = surprisal_bits_subset(logits[pred_pos], cand_ids_ext_t[:C], cidx[w_next])

    add_surprisal(events_orig, tracked_orig)
    add_surprisal(events_ext, tracked_ext)

    def agg(events, key):
        sw = np.array([e[key] for e in events if e["switch"] and key in e])
        st = np.array([e[key] for e in events if not e["switch"] and key in e])
        return (float(sw.mean()) if sw.size else None), (float(st.mean()) if st.size else None)

    supply_sw, supply_st = agg(events_orig, "supply")
    supply_ext_sw, supply_ext_st = agg(events_ext, "supply_ext")
    surp_full_sw, surp_full_st = agg(events_orig, "surprisal_full_bits")
    surp_cand_sw, surp_cand_st = agg(events_orig, "surprisal_candset_bits")
    surp_full_ext_sw, surp_full_ext_st = agg(events_ext, "surprisal_full_bits")
    tcs_true_sw, tcs_true_st = agg(events_ext, "total_category_size_true")
    n_switch_orig = sum(1 for e in events_orig if e["switch"])
    n_switch_ext = sum(1 for e in events_ext if e["switch"])

    return {
        "seed": seed,
        "generated": gen_text,
        "n_switch_events": n_switch_orig, "n_stay_events": len(events_orig) - n_switch_orig,
        "n_switch_events_ext": n_switch_ext, "n_stay_events_ext": len(events_ext) - n_switch_ext,
        "mean_supply_before_switch": supply_sw, "mean_supply_before_stay": supply_st,
        "mean_supply_ext_before_switch": supply_ext_sw, "mean_supply_ext_before_stay": supply_ext_st,
        "mean_surprisal_full_bits_switch": surp_full_sw, "mean_surprisal_full_bits_stay": surp_full_st,
        "mean_surprisal_candset_bits_switch": surp_cand_sw, "mean_surprisal_candset_bits_stay": surp_cand_st,
        "mean_surprisal_full_bits_switch_ext": surp_full_ext_sw, "mean_surprisal_full_bits_stay_ext": surp_full_ext_st,
        "mean_total_category_size_true_switch": tcs_true_sw, "mean_total_category_size_true_stay": tcs_true_st,
        "events_orig": events_orig,
        "events_ext": events_ext,
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
    word2cats   = norms["word2cats"]
    categories  = norms["categories"]
    probe_ids   = norms["probe_ids"]
    all_members = norms["all_members"]
    probe_words = list(probe_ids.keys())
    cand_ids    = [probe_ids[w] for w in probe_words]
    cidx        = {w: k for k, w in enumerate(probe_words)}
    word_of_id  = N.build_word_of_id(tok, probe_ids)
    C = len(probe_words)

    leading_groups = N.build_leading_token_groups(word2cats, probe_ids, tok)
    # indexed by every recognised word, since multi-token words can be event origins
    pool_ext, leading_ids_sorted = N.build_patch_pool_ext(word2cats, categories, all_members, cidx, leading_groups)
    G = len(leading_ids_sorted)
    leading_cache = {}   # memo for word detection, shared across seeds
    cand_ids_ext = cand_ids + leading_ids_sorted   # [0:C) single-token, [C:C+G) leading-token groups

    n_multi_words = sum(len(v) for v in leading_groups.values())
    print(f"layer {layer}: {C} single-token probes, {G} leading-token groups covering "
          f"{n_multi_words} multi-token animals, top_k={top_k}, "
          f"seeds {seed_start}-{seed_start + n_seeds - 1}", flush=True)

    patch_pool = {}
    for w in probe_words:
        nbrs = N.within_patch_members(w, word2cats, categories)
        patch_pool[w] = [cidx[m] for m in nbrs if m in cidx]

    cand_ids_ext_t = torch.as_tensor(cand_ids_ext, device=J.DEVICE)
    jlens_pt = os.environ.get("JLENS_PT")
    seed_end = seed_start + n_seeds - 1
    layer_tag = f"{layer:02d}_seed{seed_start:03d}-{seed_end:03d}"

    V, Jmat = None, None
    if jlens_pt:
        Jmat = J.load_jlens_pt(jlens_pt)["J"]
        W_U, _ = J.get_unembed_and_norm(model)
        V_lens = torch.nn.functional.normalize(
            W_U[cand_ids_ext_t].float() @ Jmat[layer].to(J.DEVICE, torch.float32), dim=1)
        np.save(os.path.join(out_dir, f"jlens_V_ext_layer{layer_tag}.npy"), V_lens.cpu().numpy())
    else:
        corpus = J.load_corpus(120, os.environ.get("JLENS_CORPUS"))
        V = J.compute_jlens_vectors(model, tok, cand_ids_ext, layer, corpus)
        np.save(os.path.join(out_dir, f"jlens_V_ext_layer{layer_tag}.npy"), V.cpu().numpy())

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
                                     cand_ids_ext, cand_ids_ext_t, C, probe_words, cidx, word_of_id,
                                     word2cats, categories, all_members, patch_pool, pool_ext, leading_cache,
                                     seed, top_k=top_k))

    out = {
        "layer": layer, "top_k": top_k,
        "probe_words": probe_words,               # single-token block, row order matches jlens_V_ext[:C]
        "n_leading_groups": G, "n_multi_token_words": n_multi_words,
        "used_precomputed_lens": bool(jlens_pt),
        "jlens_pt": jlens_pt,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "records": records,
    }
    with open(os.path.join(out_dir, f"exp2ext_layer{layer_tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    def mean_of(field):
        vals = [r[field] for r in records if r[field] is not None]
        return float(np.mean(vals)) if vals else None

    print(json.dumps({
        "layer": layer, "n_seeds": n_seeds,
        "total_switch_events_orig": sum(r["n_switch_events"] for r in records),
        "total_events_orig": sum(r["n_switch_events"] + r["n_stay_events"] for r in records),
        "total_switch_events_ext": sum(r["n_switch_events_ext"] for r in records),
        "total_events_ext": sum(r["n_switch_events_ext"] + r["n_stay_events_ext"] for r in records),
        "mean_of_per_seed_mean_supply_switch": mean_of("mean_supply_before_switch"),
        "mean_of_per_seed_mean_supply_stay": mean_of("mean_supply_before_stay"),
        "mean_of_per_seed_mean_supply_ext_switch": mean_of("mean_supply_ext_before_switch"),
        "mean_of_per_seed_mean_supply_ext_stay": mean_of("mean_supply_ext_before_stay"),
    }, indent=2))


if __name__ == "__main__":
    main()
