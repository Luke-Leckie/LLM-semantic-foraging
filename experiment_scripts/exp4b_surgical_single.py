# exp4b_surgical_single.py
# Usage: python3 exp4b_surgical_single.py <layer> <model_path> <out_dir> <norms_csv> <seed_start> <n_seeds> [steer_scales]
#
# Experiment 4b (surgical): does steering along a category LABEL WORD's J-lens
# direction (e.g. "water") at a single mid-patch branch point redirect the next
# item into that category?
#   * Per seed, generate a baseline rollout. For each target category, sample one
#     branch point: an item that continues the previous item's category and is not
#     itself in the target category.
#   * From that prefix, generate a short continuation under three conditions:
#     baseline (the original rollout's own continuation), steer (label direction
#     added at `layer`) and noise (random direction orthogonal to it, same norm).
#   * Record whether the first new item is in the target category (immediate) and
#     whether any item in the continuation is (any).
# Several label words share a target category (e.g. water/sea); they share the
# same branch point, so differences between them reflect the word itself. Labels
# that are not a single token are dropped.
#
# steer_scales: comma-separated multiples of the typical residual norm at `layer`
# (default 1.0,0.5,0.25,0.1). Saves raw hit/total counts; rates and statistics are
# computed downstream.
#
# Output: <out_dir>/exp4bsurg_layer<LL>_seed<start>-<end>_scale<scales>.json
import sys, os, json, random, torch
from collections import defaultdict
import jlens_common as J
import jlens_norms as N

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
]

STEER_SCALES = [1.0, 0.5, 0.25, 0.1]   # default; overridable via CLI
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


def get_label_ids(tok, label_words):
    """({label_word: token_id} for single-token labels, [dropped multi-token labels])."""
    label_ids, dropped = {}, []
    for word in label_words:
        ids = tok.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            label_ids[word] = ids[0]
        else:
            dropped.append(word)
    return label_ids, dropped


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
    word_of_id = N.build_word_of_id(tok, probe_ids)

    label_words = sorted({label for label, _ in LABEL_TARGETS})
    label_ids, dropped_labels = get_label_ids(tok, label_words)
    # one result bucket per surviving pair, keyed "{label}->{target_cat}"
    pairs = [(label, target) for label, target in LABEL_TARGETS if label in label_ids]
    pair_keys = [f"{label}->{target}" for label, target in pairs]
    dropped_pairs = [f"{label}->{target}" for label, target in LABEL_TARGETS if label not in label_ids]
    if dropped_labels:
        print(f"NOTE: labels {dropped_labels} are not single-token in this tokenizer -- "
              f"dropped entirely (pairs affected: {dropped_pairs})", flush=True)

    # branch points are sampled once per (seed, target_cat) and shared by all its labels
    labels_by_target = defaultdict(list)
    for label, target in pairs:
        labels_by_target[target].append(label)
    target_cats = sorted(labels_by_target.keys())
    print(f"layer {layer}: {len(pairs)} label->target pairs over {len(target_cats)} target "
          f"categories: {dict(labels_by_target)}, seeds {seed_start}-{seed_start + n_seeds - 1}",
          flush=True)

    jlens_pt = os.environ.get("JLENS_PT")
    # probe vectors for single-token labels only
    survivor_words = sorted(label_ids.keys())
    cand_ids = [label_ids[w] for w in survivor_words]
    V = J.get_probe_vectors(model, tok, layer, cand_ids, jlens_pt=jlens_pt)
    vidx = {w: k for k, w in enumerate(survivor_words)}

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
    results = {str(scale): {key: {"n_branch_points": 0,
                                   "baseline": dict(EMPTY), "steer": dict(EMPTY), "noise": dict(EMPTY)}
                             for key in pair_keys}
               for scale in steer_scales}
    # stable per-label index for seeding (Python's str hash is not reproducible)
    label_to_idx = {w: i for i, w in enumerate(survivor_words)}

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
        for target_cat in target_cats:
            branch_idxs = find_branch_points(tracked, word2cats, target_cat)
            if not branch_idxs:
                continue
            i = rng.choice(branch_idxs)
            branch_pos = tracked[i][0]
            prefix_ids = full_ids[:branch_pos + 1].unsqueeze(0)
            prefix_attn = torch.ones_like(prefix_ids)
            prefix_len = prefix_ids.shape[1]

            # baseline: the original rollout's own continuation from this point
            baseline_ids = gen_ids[branch_pos - prompt_len + 1:]
            baseline_text = tok.decode(torch.tensor(baseline_ids), skip_special_tokens=True)
            b_imm, b_any = continuation_hits(baseline_ids, baseline_text, prefix_len,
                                              word_of_id, word2cats, target_cat)

            for scale in steer_scales:
                for label in labels_by_target[target_cat]:
                    key = f"{label}->{target_cat}"
                    v = V[vidx[label]]
                    noise_dir = J.random_orthogonal_unit(v, generator=noise_rng)
                    steer_vec = scale * norm_scale * v
                    noise_vec = scale * norm_scale * noise_dir

                    label_tag = label_to_idx[label]
                    for cond, vec, off in (("steer", steer_vec, 4000000), ("noise", noise_vec, 5000000)):
                        handle = J.register_additive_hook(model, layer, vec)
                        try:
                            torch.manual_seed(off + label_tag * 1000 + seed)
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
                                                       word_of_id, word2cats, target_cat)
                        acc = results[str(scale)][key][cond]
                        if imm is not None:
                            acc["immediate_total"] += 1
                            acc["immediate_hits"] += int(imm)
                        if any_ is not None:
                            acc["any_total"] += 1
                            acc["any_hits"] += int(any_)

                    acc_b = results[str(scale)][key]["baseline"]
                    if b_imm is not None:
                        acc_b["immediate_total"] += 1
                        acc_b["immediate_hits"] += int(b_imm)
                    if b_any is not None:
                        acc_b["any_total"] += 1
                        acc_b["any_hits"] += int(b_any)
                    results[str(scale)][key]["n_branch_points"] += 1

    seed_end = seed_start + n_seeds - 1
    layer_tag = f"{layer:02d}_seed{seed_start:03d}-{seed_end:03d}"
    # scales in the filename so runs with different scale sets don't overwrite each other
    scale_tag = "-".join(f"{float(s):g}" for s in steer_scales)
    out = {
        "layer": layer,
        "used_precomputed_lens": bool(jlens_pt),
        "jlens_pt": jlens_pt,
        "label_targets": LABEL_TARGETS,
        "dropped_labels": dropped_labels,
        "steer_scales": steer_scales,
        "gen_max_new_tokens_continuation": GEN_MAX_NEW_TOKENS_CONTINUATION,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "results": results,
    }
    with open(os.path.join(out_dir, f"exp4bsurg_layer{layer_tag}_scale{scale_tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    n_bp = {key: results[str(steer_scales[0])][key]["n_branch_points"] for key in pair_keys}
    print(json.dumps({"layer": layer, "n_seeds": n_seeds, "branch_points_used_per_pair": n_bp}, indent=2))


if __name__ == "__main__":
    main()
