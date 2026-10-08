# exp4c_surgical_steering_single.py
# Usage: python3 exp4c_surgical_steering_single.py <layer> <model_path> <out_dir> <norms_csv> <seed_start> <n_seeds> [steer_scales]
#
# Experiment 4c (surgical, item-level): can steering at a single decision point
# redirect the next item into a target category T?
#   1. Generate one unsteered baseline rollout per seed.
#   2. Per category T, sample one branch point: an item that is adjacent to the
#      previous item, continues its category (mid-patch), and is not in T.
#   3. From that prefix, generate a short continuation with T's direction added at
#      `layer` (steer) and with a random direction orthogonal to it, same norm
#      (noise). The baseline is the original rollout's own continuation.
#   4. Record whether the first new item is in T (immediate) and whether any item
#      in the continuation is (any).
# T's direction is the normalised mean of the J-lens directions of T's
# single-token members. Categories with no branch point in a seed are skipped.
#
# steer_scales: comma-separated multiples of the typical residual norm at `layer`
# (default 0.5). Saves raw hit/total counts; rates and statistics are computed
# downstream.
#
# Output: <out_dir>/exp4c_layer<LL>_seed<start>-<end>.json
import sys, os, json, random, torch
import jlens_common as J
import jlens_norms as N

STEER_SCALES = [0.5]   # default; overridable via CLI
GEN_MAX_NEW_TOKENS_BASELINE = 220
GEN_MAX_NEW_TOKENS_CONTINUATION = 40   # only the next few items are scored
TEMP, TOP_P = 0.9, 0.95


def find_branch_points(tracked, word2cats, target_cat):
    """Indices i (>= 1) into `tracked` where item i is adjacent to item i-1, continues
    its category (mid-patch), and is not in target_cat."""
    out = []
    for i in range(1, len(tracked)):
        _, w, gap = tracked[i]
        _, w_prev, _ = tracked[i - 1]
        if gap:
            continue
        if N.is_switch(w_prev, w, word2cats):
            continue
        if target_cat in word2cats.get(w, set()):
            continue
        out.append(i)
    return out


def continuation_hits(cont_ids, cont_text, prefix_len, word_of_id, word2cats, target_cat):
    """(first new item is in target_cat, any item is in target_cat); (None, None)
    if the continuation contains no tracked item."""
    tracked = N.build_tracked_sequence(cont_text, cont_ids, prefix_len, word_of_id)
    items = [w for _, w, gap in tracked if not gap]
    if not items:
        return None, None
    immediate_hit = target_cat in word2cats.get(items[0], set())
    any_hit = any(target_cat in word2cats.get(w, set()) for w in items)
    return immediate_hit, any_hit


def main():
    layer      = int(sys.argv[1])
    model_path = sys.argv[2]
    out_dir    = sys.argv[3]
    norms_csv  = sys.argv[4]
    seed_start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    n_seeds    = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    steer_scales = ([float(x) for x in sys.argv[7].split(",")]
                    if len(sys.argv) > 7 else STEER_SCALES)
    os.makedirs(out_dir, exist_ok=True)

    noise_rng = torch.Generator(device=J.DEVICE).manual_seed(0)

    model, tok = J.load_model(model_path)
    norms = N.load_troyer_norms(norms_csv, tok, verbose=False)
    word2cats, categories, probe_ids = norms["word2cats"], norms["categories"], norms["probe_ids"]
    probe_words = list(probe_ids.keys())
    cand_ids = [probe_ids[w] for w in probe_words]
    cidx = {w: k for k, w in enumerate(probe_words)}
    word_of_id = N.build_word_of_id(tok, probe_ids)
    cat_names = sorted(categories.keys())
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}   # stable index for seeding
    print(f"layer {layer}: {len(cat_names)} categories, seeds {seed_start}-{seed_start + n_seeds - 1}",
          flush=True)

    jlens_pt = os.environ.get("JLENS_PT")
    V = J.get_probe_vectors(model, tok, layer, cand_ids, jlens_pt=jlens_pt)

    full_ids0, _ = J.fluency_generate(model, tok)
    norm_scale = J.typical_residual_norm(model, tok, full_ids0, layer)
    print(f"typical residual norm at layer {layer}: {norm_scale:.2f}", flush=True)
    print(f"steer scales: {steer_scales}", flush=True)

    prompt = ("Name as many different animals as you can, one after another, "
              "separated by commas. Just the list.")
    msgs = [{"role": "user", "content": prompt}]
    gen_in = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(gen_in):
        gen_in = gen_in["input_ids"]   # newer transformers can return a BatchEncoding here
    gen_in = gen_in.to(J.DEVICE)
    attn_mask = torch.ones_like(gen_in)
    prompt_len = gen_in.shape[1]

    EMPTY = {"immediate_hits": 0, "immediate_total": 0, "any_hits": 0, "any_total": 0}
    results = {str(scale): {cat: {"n_branch_points": 0,
                                   "baseline": dict(EMPTY), "steer": dict(EMPTY), "noise": dict(EMPTY)}
                             for cat in cat_names}
               for scale in steer_scales}

    for seed in range(seed_start, seed_start + n_seeds):
        print(f"  seed {seed}: generating baseline rollout ...", flush=True)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(gen_in, attention_mask=attn_mask,
                                  max_new_tokens=GEN_MAX_NEW_TOKENS_BASELINE, do_sample=True,
                                  temperature=TEMP, top_p=TOP_P, pad_token_id=tok.eos_token_id)
        full_ids = out[0]
        gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)
        gen_ids = full_ids[prompt_len:].tolist()
        tracked = N.build_tracked_sequence(gen_text, gen_ids, prompt_len, word_of_id)
        if len(tracked) < 2:
            continue

        rng = random.Random(seed)
        for cat in cat_names:
            branch_idxs = find_branch_points(tracked, word2cats, cat)
            if not branch_idxs:
                continue
            i = rng.choice(branch_idxs)
            branch_pos = tracked[i][0]                       # abs token pos of the branch word
            prefix_ids = full_ids[:branch_pos + 1].unsqueeze(0)
            prefix_attn = torch.ones_like(prefix_ids)
            prefix_len = prefix_ids.shape[1]

            # baseline: the original rollout's own continuation from this point
            baseline_ids = gen_ids[branch_pos - prompt_len + 1:]
            baseline_text = tok.decode(torch.tensor(baseline_ids), skip_special_tokens=True)
            b_imm, b_any = continuation_hits(baseline_ids, baseline_text, prefix_len,
                                              word_of_id, word2cats, cat)

            for scale in steer_scales:
                # category direction: normalised mean of its single-token members' directions
                members = [w for w in categories[cat] if w in cidx]
                if not members:
                    continue
                vecs = torch.stack([V[cidx[w]] for w in members])
                direction = torch.nn.functional.normalize(vecs.mean(dim=0), dim=0)
                noise_dir = J.random_orthogonal_unit(direction, generator=noise_rng)
                steer_vec = scale * norm_scale * direction
                noise_vec = scale * norm_scale * noise_dir

                cat_tag = cat_to_idx[cat]
                for cond, vec, off in (("steer", steer_vec, 4000000), ("noise", noise_vec, 5000000)):
                    handle = J.register_additive_hook(model, layer, vec)
                    try:
                        torch.manual_seed(off + cat_tag * 1000 + seed)
                        with torch.no_grad():
                            cont_out = model.generate(
                                prefix_ids, attention_mask=prefix_attn,
                                max_new_tokens=GEN_MAX_NEW_TOKENS_CONTINUATION, do_sample=True,
                                temperature=TEMP, top_p=TOP_P, pad_token_id=tok.eos_token_id)
                    finally:
                        handle.remove()
                    cont_ids = cont_out[0, prefix_len:].tolist()
                    cont_text = tok.decode(cont_out[0, prefix_len:], skip_special_tokens=True)
                    imm, any_ = continuation_hits(cont_ids, cont_text, prefix_len,
                                                   word_of_id, word2cats, cat)
                    acc = results[str(scale)][cat][cond]
                    if imm is not None:
                        acc["immediate_total"] += 1
                        acc["immediate_hits"] += int(imm)
                    if any_ is not None:
                        acc["any_total"] += 1
                        acc["any_hits"] += int(any_)

                acc_b = results[str(scale)][cat]["baseline"]
                if b_imm is not None:
                    acc_b["immediate_total"] += 1
                    acc_b["immediate_hits"] += int(b_imm)
                if b_any is not None:
                    acc_b["any_total"] += 1
                    acc_b["any_hits"] += int(b_any)
                results[str(scale)][cat]["n_branch_points"] += 1

    seed_end = seed_start + n_seeds - 1
    layer_tag = f"{layer:02d}_seed{seed_start:03d}-{seed_end:03d}"
    out = {
        "layer": layer,
        "used_precomputed_lens": bool(jlens_pt),
        "jlens_pt": jlens_pt,
        "steer_scales": steer_scales,
        "gen_max_new_tokens_continuation": GEN_MAX_NEW_TOKENS_CONTINUATION,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "results": results,
    }
    with open(os.path.join(out_dir, f"exp4c_layer{layer_tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    n_bp = sum(v["n_branch_points"] for v in results[str(steer_scales[0])].values())
    print(json.dumps({"layer": layer, "n_seeds": n_seeds, "total_branch_points_used": n_bp}, indent=2))


if __name__ == "__main__":
    main()
