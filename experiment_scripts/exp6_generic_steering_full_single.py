# exp6_generic_steering_full_single.py
# Usage: python3 exp6_generic_steering_full_single.py <layer> <model_path> <out_dir> <norms_csv> <generic_dir> <seed_start> <n_seeds> [steer_scales]
#
# Experiment 6: does steering along the generic switch direction (from
# exp5_analysis_single.py, read from <generic_dir>) change how often the model
# switches category during animal fluency?
#   * baseline: unsteered rollouts;
#   * steer: scale * (typical residual norm at `layer`) * direction, added to the
#     residual stream at `layer` at every position;
#   * noise: the same, with a random unit direction orthogonal to the steering one.
# Outcome per condition: number of switches over adjacent transitions between
# recognised animals (single- or multi-token), plus zero-emission counts as a
# check that steering hasn't broken generation.
#
# steer_scales: comma-separated (default 0.5,-0.5); positive pushes towards
# switching, negative towards clustering.
#
# Output: <out_dir>/exp6_layer<LL>_seed<start>-<end>.json
import sys, os, json, torch
import numpy as np
import jlens_common as J
import jlens_norms as N

STEER_SCALES = [0.5, -0.5]   # default; overridable via CLI
GEN_MAX_NEW_TOKENS = 220
TEMP, TOP_P = 0.9, 0.95
SEED = 0   # fixed so the noise direction is identical across seed chunks


def load_generic_direction(generic_dir, layer, device):
    fp = os.path.join(generic_dir, f"generic_switch_direction_layer{layer:02d}.npy")
    direction = np.load(fp)
    v = torch.from_numpy(direction).to(device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(v, dim=0)


def run_rollouts(model, tok, gen_in, attn_mask, layer, vector, all_members, leading_cache,
                  seed_offset, seed_start, n_seeds):
    handle = J.register_additive_hook(model, layer, vector) if vector is not None else None
    try:
        all_tracked, all_generated = [], []
        for s in range(seed_start, seed_start + n_seeds):
            torch.manual_seed(seed_offset + s)
            with torch.no_grad():
                out = model.generate(gen_in, attention_mask=attn_mask,
                                      max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                                      temperature=TEMP, top_p=TOP_P, pad_token_id=tok.eos_token_id)
            full_ids = out[0]
            prompt_len = gen_in.shape[1]
            gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)
            gen_ids = full_ids[prompt_len:].tolist()
            all_tracked.append(N.build_tracked_sequence_full(gen_text, gen_ids, prompt_len,
                                                               all_members, tok, cache=leading_cache))
            all_generated.append(gen_text)
        return all_tracked, all_generated
    finally:
        if handle is not None:
            handle.remove()


def switch_counts(all_tracked, all_generated, word2cats):
    """Switch / transition / emission counts over a condition's rollouts, plus the
    number of rollouts with no recognised animal. Raw texts are kept for inspection."""
    n_switches, n_transitions, n_emissions, n_zero_emission_seeds = 0, 0, 0, 0
    for tracked in all_tracked:
        seq = [(pos, w) for pos, w, _ in tracked]
        n_emissions += len(seq)
        if len(seq) == 0:
            n_zero_emission_seeds += 1
        for n in range(len(seq) - 1):
            if tracked[n + 1][2]:
                continue   # not adjacent (unrecognised item in between)
            w, w_next = seq[n][1], seq[n + 1][1]
            n_transitions += 1
            if N.is_switch(w, w_next, word2cats):
                n_switches += 1
    return {"n_switches": n_switches, "n_transitions": n_transitions,
            "n_emissions": n_emissions, "n_zero_emission_seeds": n_zero_emission_seeds,
            "n_seeds": len(all_tracked), "generated": all_generated}


def main():
    layer       = int(sys.argv[1])
    model_path  = sys.argv[2]
    out_dir     = sys.argv[3]
    norms_csv   = sys.argv[4]
    generic_dir = sys.argv[5]
    seed_start  = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    n_seeds     = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    steer_scales = ([float(x) for x in sys.argv[8].split(",")]
                    if len(sys.argv) > 8 else STEER_SCALES)
    os.makedirs(out_dir, exist_ok=True)

    noise_rng = torch.Generator(device=J.DEVICE).manual_seed(SEED)

    model, tok = J.load_model(model_path)
    norms = N.load_troyer_norms(norms_csv, tok, verbose=False)
    word2cats = norms["word2cats"]
    all_members = norms["all_members"]
    leading_cache = {}   # memo for word detection, shared across rollouts

    direction = load_generic_direction(generic_dir, layer, J.DEVICE)
    noise_dir = J.random_orthogonal_unit(direction, generator=noise_rng)
    print(f"layer {layer}: loaded generic_switch_direction (dim={direction.shape[0]}), "
          f"seeds {seed_start}-{seed_start + n_seeds - 1}", flush=True)

    full_ids, _ = J.fluency_generate(model, tok)
    norm_scale = J.typical_residual_norm(model, tok, full_ids, layer)
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

    # baseline is independent of scale, so generate it once
    print("generating baseline rollouts ...", flush=True)
    baseline_tracked, baseline_gen = run_rollouts(model, tok, gen_in, attn_mask, layer, None,
                                                   all_members, leading_cache, seed_offset=1000000,
                                                   seed_start=seed_start, n_seeds=n_seeds)
    baseline_counts = switch_counts(baseline_tracked, baseline_gen, word2cats)
    print("  [baseline] sample rollouts:", flush=True)
    for gt in baseline_gen[:2]:
        print(f"    seed sample: {gt[:200]!r}", flush=True)

    results = {}
    for scale in steer_scales:
        print(f"scale={scale}: generating steer/noise rollouts ...", flush=True)
        steer_vec = scale * norm_scale * direction
        noise_vec = scale * norm_scale * noise_dir

        steer_tracked, steer_gen = run_rollouts(model, tok, gen_in, attn_mask, layer, steer_vec,
                                                 all_members, leading_cache, seed_offset=2000000,
                                                 seed_start=seed_start, n_seeds=n_seeds)
        noise_tracked, noise_gen = run_rollouts(model, tok, gen_in, attn_mask, layer, noise_vec,
                                                 all_members, leading_cache, seed_offset=3000000,
                                                 seed_start=seed_start, n_seeds=n_seeds)

        results[str(scale)] = {
            "baseline": baseline_counts,
            "steer": switch_counts(steer_tracked, steer_gen, word2cats),
            "noise": switch_counts(noise_tracked, noise_gen, word2cats),
        }

        # print a couple of rollouts per condition to check generation is still coherent
        for cond, gen_list in (("steer", steer_gen), ("noise", noise_gen)):
            print(f"  scale={scale} [{cond}] sample rollouts:", flush=True)
            for gt in gen_list[:2]:
                print(f"    seed sample: {gt[:200]!r}", flush=True)

    seed_end = seed_start + n_seeds - 1
    layer_tag = f"{layer:02d}_seed{seed_start:03d}-{seed_end:03d}"
    out = {
        "layer": layer,
        "generic_dir": generic_dir,
        "steer_scales": steer_scales,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "results": results,
    }
    with open(os.path.join(out_dir, f"exp6_layer{layer_tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
