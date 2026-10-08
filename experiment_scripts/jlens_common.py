# jlens_common.py
# ---------------------------------------------------------------------
# Shared Jacobian-lens machinery for Gemma-2 (softcap-aware).
# The J-lens vector for token t at layer L is the averaged sensitivity of
# token t's final-layer logit to the residual stream at layer L:
#     v_t = E_ctx[ d logit_t / d h_L ]  (a VJP; no full Jacobian needed).
# Causality gives the t' >= t masking for free.
# Gemma-2 applies final-logit softcapping, so we differentiate the
# PRE-softcap logit, norm(h_final) @ W_U, not model.logits.
# ---------------------------------------------------------------------
import os, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small generic "animal" probe list used by Experiment 1 when NOT using the
# Troyer norms. For the norms-driven runs, probes come from jlens_norms instead.
ANIMAL_WORDS = """cat dog wolf fox bear lion tiger horse cow pig sheep goat deer
mouse rat rabbit hare squirrel bat frog toad snake lizard turtle shark whale
dolphin seal otter crab lobster shrimp octopus squid eagle hawk owl crow raven
duck goose swan chicken turkey pigeon sparrow robin penguin ant bee wasp moth
spider beetle fly worm snail slug camel donkey monkey ape gorilla zebra giraffe
elephant rhino hippo kangaroo koala panda leopard cheetah panther jaguar""".split()

# Averaging background: GENERIC, non-animal, pretraining-like text. This is the
# distribution the Jacobian is averaged over, NOT what we probe for. Offline on
# the cluster load_dataset fails, so point JLENS_CORPUS at a transferred text
# file, or expand this inline list to a few hundred varied sentences.
FALLBACK_CORPUS = [
    "The history of the region is complex and spans many centuries.",
    "In physics, energy is conserved across an isolated system.",
    "The committee met on Tuesday to review the annual budget report.",
    "Rainfall patterns shifted noticeably over the last several decades.",
    "She opened the heavy book and began to read the first chapter.",
    "Economic policy in the postwar era favored broad public investment.",
    "The telescope revealed faint galaxies near the edge of the field.",
    "Farmers rotated their crops to keep the soil healthy each season.",
    "The orchestra tuned quietly before the conductor raised her baton.",
    "Software engineers debated the merits of the new caching layer.",
]


def load_model(model_path, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager"
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


_LAYERS_PATHS = ["model.layers", "model.language_model.layers", "language_model.model.layers",
                 "language_model.layers"]
_NORM_PATHS = ["model.norm", "model.language_model.norm", "language_model.model.norm",
               "language_model.norm"]


def _resolve_attr_path(model, path):
    obj = model
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def get_decoder_layers(model):
    """Return the model's decoder-layer ModuleList, robust to multimodal wrapper
    architectures. Every plain text-only causal LM here (Gemma-2, Llama, Qwen2) has
    decoder blocks at `model.model.layers` -- tried FIRST, so behavior for every
    model already in use is unchanged. Gemma-3's checkpoints (google/gemma-3-*-it)
    are `Gemma3ForConditionalGeneration`, a multimodal wrapper with the actual text
    backbone nested one level deeper (`model.model.language_model.layers`, per its
    HF config's separate "text_config") -- confirmed from the published config
    structure, NOT from an actual load in this repo (no GPU access here to test) --
    verify this resolves correctly for gemma-3-12b-it before trusting a full run;
    if the exact attribute name differs, add it to _LAYERS_PATHS above."""
    for path in _LAYERS_PATHS:
        result = _resolve_attr_path(model, path)
        if result is not None:
            return result
    raise AttributeError(
        f"get_decoder_layers: none of {_LAYERS_PATHS} resolved on a {type(model).__name__} -- "
        f"inspect model.named_modules() and add the correct path to _LAYERS_PATHS in jlens_common.py")


def get_final_norm(model):
    """Return the model's final RMSNorm module -- same multimodal-wrapper caveat as
    get_decoder_layers above."""
    for path in _NORM_PATHS:
        result = _resolve_attr_path(model, path)
        if result is not None:
            return result
    raise AttributeError(
        f"get_final_norm: none of {_NORM_PATHS} resolved on a {type(model).__name__} -- "
        f"inspect model.named_modules() and add the correct path to _NORM_PATHS in jlens_common.py")


def get_num_hidden_layers(model):
    """True decoder layer count, robust to multimodal wrapper architectures -- same
    caveat as get_decoder_layers: `model.config.num_hidden_layers` is flat for every
    text-only model here, but Gemma-3's config nests it under a separate
    `text_config` (per its published HF config structure)."""
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
        return cfg.text_config.num_hidden_layers
    raise AttributeError(
        f"get_num_hidden_layers: no num_hidden_layers on {type(cfg).__name__} directly or "
        f"under .text_config -- inspect model.config and adjust jlens_common.py")


def get_unembed_and_norm(model):
    """Return (W_U [vocab, D], final_norm_module). Handles tied embeddings."""
    W_U = model.get_output_embeddings().weight          # [vocab, D]
    final_norm = get_final_norm(model)                  # Gemma2RMSNorm / equivalent
    return W_U, final_norm


def single_token_candidates(tok, words=ANIMAL_WORDS):
    """Keep words that are a single leading-space token (their in-list form)."""
    cand = {}
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            cand[w] = ids[0]
    return cand


def load_corpus(n, path=None):
    if path and os.path.exists(path):
        with open(path) as f:
            lines = [ln.strip() for ln in f if len(ln.strip()) > 80]
        if lines:
            return lines[:n]
    reps = n // len(FALLBACK_CORPUS) + 1
    return (FALLBACK_CORPUS * reps)[:n]


def compute_jlens_vectors(model, tok, cand_ids, layer, corpus,
                          avg_seq_len=48, softcap_bypass=True, verbose=True):
    """
    Averaged J-lens vectors for candidate tokens at `layer`.
    Returns V_unit [C, D] (unit-normalized rows).
    """
    W_U, final_norm = get_unembed_and_norm(model)
    D = model.config.hidden_size
    C = len(cand_ids)
    cand_ids = torch.as_tensor(cand_ids, device=DEVICE)
    U = W_U[cand_ids].float()                            # [C, D] unembed rows
    embed = model.get_input_embeddings()
    V = torch.zeros(C, D, device=DEVICE, dtype=torch.float32)
    n_used = 0

    for i, text in enumerate(corpus):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=avg_seq_len).input_ids.to(DEVICE)
        if ids.shape[1] < 4:
            continue
        inp = embed(ids).detach().requires_grad_(True)   # forces graph to build
        out = model(inputs_embeds=inp, output_hidden_states=True, use_cache=False)
        h_L = out.hidden_states[layer + 1]               # residual after layer `layer`
        last = out.hidden_states[-1]                     # pre-final-norm last layer

        if softcap_bypass:
            normed = final_norm(last)[0].float()         # [S, D]
            cand_logits = normed @ U.T                   # [S, C] pre-softcap
        else:
            cand_logits = out.logits[0].float()[:, cand_ids]

        for j in range(C):
            obj = cand_logits[:, j].sum()
            g = torch.autograd.grad(obj, h_L, retain_graph=(j < C - 1))[0][0]
            V[j] += g.mean(dim=0).float()
        n_used += 1
        if verbose and (i + 1) % 20 == 0:
            print(f"  averaged {n_used} prompts", flush=True)

    V /= max(n_used, 1)
    return F.normalize(V, dim=1)                          # [C, D]


def get_probe_vectors(model, tok, layer, cand_ids, jlens_pt=None, corpus_n=120):
    """Unit J-lens direction vectors for `cand_ids` at `layer`: from a
    precomputed jacobian-lens checkpoint if `jlens_pt` is given (pass
    os.environ.get("JLENS_PT") at call sites, by this repo's convention),
    else corpus-averaged gradients (JLENS_CORPUS env var, or the fallback
    corpus). Returns V [C, D] unit-normalized, row order matching cand_ids."""
    if jlens_pt:
        ckpt = load_jlens_pt(jlens_pt)
        W_U, _ = get_unembed_and_norm(model)
        cand_ids_t = torch.as_tensor(cand_ids, device=DEVICE)
        return F.normalize(
            W_U[cand_ids_t].float() @ ckpt["J"][layer].to(DEVICE, torch.float32), dim=1)
    corpus = load_corpus(corpus_n, os.environ.get("JLENS_CORPUS"))
    return compute_jlens_vectors(model, tok, cand_ids, layer, corpus)


def read_activations(model, tok, full_ids, layer, return_logits=False, normalize=True):
    """Residual stream at `layer` for every position, [S, D] float32.
    Unit-normalized by default (for cosine-similarity probe scoring); pass
    normalize=False to get the raw residual (needed to transport through a
    precomputed Jacobian lens, which was fit on the real, unnormalized
    residual stream). If return_logits, also return the (post-softcap)
    logits used for generation, [S, vocab] float32 -- needed for surprisal,
    not gradients."""
    with torch.no_grad():
        out = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
    H = out.hidden_states[layer + 1][0].float()          # [S, D]
    if normalize:
        H = F.normalize(H, dim=1)
    if return_logits:
        return H, out.logits[0].float()                   # [S, vocab]
    return H


def load_jlens_pt(path):
    """Load a precomputed Anthropic jacobian-lens checkpoint
    (https://github.com/anthropics/jacobian-lens), as produced by Neuronpedia's
    fit_lens.py. Format: {"J": {layer: [D,D] tensor}, "n_prompts", "d_model",
    "source_layers"}. J[l] is the averaged Jacobian transporting the residual
    at layer l forward to the final-layer residual: h_final ~= J[l] @ h_l."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in ckpt:
        raise ValueError(f"{path}: no 'J' key -- not a jacobian-lens checkpoint "
                          f"(found keys {sorted(ckpt)!r})")
    return ckpt


def lens_forward_logits(model, H_raw, layer, J, cand_ids):
    """Forward-lens candidate logits at every position, using a precomputed
    Jacobian lens instead of a corpus-averaged probe direction: transport the
    REAL residual `H_raw` [S, D] (raw, NOT unit-normalized) forward through
    J[layer] to an estimated final-layer residual, then apply the model's own
    final_norm + unembed to it, exactly as jlens_common's softcap-bypass
    readout does for the true final layer -- but with no backward pass and no
    corpus averaging needed at call time (J was fit once, offline).
    Returns cand_logits [S, C] float32.
    """
    W_U, final_norm = get_unembed_and_norm(model)
    cand_ids_t = torch.as_tensor(cand_ids, device=H_raw.device)
    U = W_U[cand_ids_t].float()                            # [C, D]
    J_bar = J[layer].to(device=H_raw.device, dtype=torch.float32)   # [D, D]
    h_final_est = H_raw.float() @ J_bar.T                  # [S, D], transport() convention
    normed = final_norm(h_final_est.to(model.dtype)).float()        # [S, D]
    return normed @ U.T                                     # [S, C] pre-softcap logits


def fluency_generate(model, tok, category="animals", max_new_tokens=200,
                     do_sample=False, temperature=0.9, top_p=0.95, seed=None):
    """do_sample=False (default) reproduces the original single deterministic
    trajectory. For multi-seed runs, pass do_sample=True and a seed -- matches
    the temperature/top_p convention used in exp2/exp3/exp4."""
    prompt = (f"Name as many different {category} as you can, one after another, "
              f"separated by commas. Just the list.")
    msgs = [{"role": "user", "content": prompt}]
    gen_in = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(gen_in):
        gen_in = gen_in["input_ids"]   # newer transformers can return a BatchEncoding here
    gen_in = gen_in.to(DEVICE)
    attn_mask = torch.ones_like(gen_in)
    if do_sample and seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(gen_in, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
                             do_sample=do_sample,
                             **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
                             pad_token_id=tok.eos_token_id)
    return out[0], gen_in.shape[1]


def register_additive_hook(model, layer, vector):
    """Add `vector` [D] to the residual stream output of decoder block
    `layer` on every forward call (prompt AND generated tokens alike), until
    the returned handle's .remove() is called. Uses the same block-index
    convention as the rest of this module: `layer` here == hidden_states[layer+1],
    i.e. this hooks the module whose output IS that residual.
    `vector` should already be scaled (see typical_residual_norm) -- this
    function does no normalization of its own."""
    decoder_layer = get_decoder_layers(model)[layer]

    def _hook(module, inputs, output):
        v = vector.to(device=output[0].device if isinstance(output, tuple) else output.device,
                       dtype=output[0].dtype if isinstance(output, tuple) else output.dtype)
        if isinstance(output, tuple):
            return (output[0] + v,) + output[1:]
        return output + v

    return decoder_layer.register_forward_hook(_hook)


def register_mlp_capture_hook(model, layer, storage):
    """Forward PRE-hook on decoder block `layer`'s MLP down_proj -- captures
    its INPUT, i.e. the intermediate_size-dim "neuron" activations (after
    the gate/up projections and nonlinearity, before down-projecting back to
    hidden_size). Confirmed against Gemma2MLP.forward:
        down_proj(act_fn(gate_proj(x)) * up_proj(x))
    so down_proj's input IS exactly that product -- the classic
    mech-interp sense of "neuron" for a GeGLU-style MLP, as opposed to
    register_additive_hook's residual-stream (post-block) convention.
    Appends the captured tensor (still on-device, not detached/copied -- the
    caller decides when to .float().cpu()) to `storage` on every forward
    call, until the returned handle's .remove() is called; for a single
    full-sequence forward pass this ends up as ONE entry, shape
    [1, S, intermediate_size], indexable by position like hidden_states."""
    module = get_decoder_layers(model)[layer].mlp.down_proj

    def _hook(mod, args):
        storage.append(args[0])

    return module.register_forward_pre_hook(_hook)


def register_mlp_additive_hook(model, layer, vector):
    """Add `vector` [intermediate_size] to decoder block `layer`'s MLP
    down_proj INPUT (the GeGLU "neuron" activation -- see
    register_mlp_capture_hook) on every forward call, until the returned
    handle's .remove() is called. This is the write-side counterpart of
    register_mlp_capture_hook: same module, same
    act_fn(gate_proj(x))*up_proj(x) tensor, but a pre-hook that MODIFIES the
    input in place of just recording it, so the (perturbed) MLP output flows
    into the residual stream exactly where it normally would.
    Unlike register_additive_hook, this operates in intermediate_size-dim
    space (14336 for 9B), not hidden_size/D -- `vector` must already be in
    that space and already scaled (see typical_mlp_activation_norm)."""
    module = get_decoder_layers(model)[layer].mlp.down_proj

    def _hook(mod, args):
        v = vector.to(device=args[0].device, dtype=args[0].dtype)
        return (args[0] + v,) + args[1:]

    return module.register_forward_pre_hook(_hook)


def typical_mlp_activation_norm(model, tok, full_ids, layer):
    """Mean L2 norm of the real MLP down_proj-input ("neuron") activation at
    `layer`, for scaling an MLP-space steering vector in units comparable to
    the model's own activations -- the intermediate_size-space analogue of
    typical_residual_norm."""
    storage = []
    handle = register_mlp_capture_hook(model, layer, storage)
    try:
        with torch.no_grad():
            model(full_ids.unsqueeze(0), output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    H = storage[0][0].float()   # [S, intermediate_size]
    return float(H.norm(dim=1).mean())


def typical_residual_norm(model, tok, full_ids, layer):
    """Mean L2 norm of the real residual stream at `layer`, for scaling a
    steering vector in units comparable to the model's own activations
    rather than an arbitrary fixed constant (residual norms vary a lot by
    layer and by model)."""
    with torch.no_grad():
        out = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
    H = out.hidden_states[layer + 1][0].float()
    return float(H.norm(dim=1).mean())


def random_orthogonal_unit(vec, generator=None):
    """A random unit vector in the same space as `vec` [D], with its
    component along `vec` removed (Gram-Schmidt) -- a same-dimensionality
    "noise" direction that carries no information about `vec` specifically,
    for use as a steering control condition. Pass an explicit `generator`
    (torch.Generator) to keep this independent of any torch.manual_seed()
    calls elsewhere (e.g. for reproducible generation sampling) -- note
    torch.randn (unlike torch.randn_like) is what actually accepts one."""
    r = torch.randn(vec.shape, generator=generator, device=vec.device, dtype=vec.dtype)
    r = r - (r @ vec) * vec        # vec assumed already unit-norm
    return F.normalize(r, dim=0)
