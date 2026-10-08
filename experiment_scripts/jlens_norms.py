# jlens_norms.py
# ---------------------------------------------------------------------
# Loader for the extended Troyer-style animal norms (wide CSV: one column
# per category, rows list member animals; cells may be blank; multiword
# animals use underscores, e.g. "fennec_fox").
#
# Produces two things that play DIFFERENT roles:
#   * probe set  -> single-token animals only. These become J-lens probe
#                   directions AND are the emissions we can detect in a
#                   generated list. (Multiword animals are multi-token, so
#                   the J-lens cannot surface them.)
#   * patch info -> the FULL category membership (incl. multiword animals),
#                   used to label emitted animals and define patch switches.
#
# Switch definition (standard Troyer clustering): two consecutive emitted
# animals are in the SAME patch if they share >= 1 category, and a SWITCH
# occurs when they share none.
# ---------------------------------------------------------------------
import csv, os


def _norm(word):
    """Underscore form -> lowercase surface form with spaces."""
    return word.strip().lower().replace("_", " ")


class WhitespaceTokStub:
    """Stand-in for a HF tokenizer, for callers that only need load_troyer_norms's
    tokenizer-INDEPENDENT outputs (`categories`/`word2cats`) and have no model loaded
    at all -- e.g. exp5_analysis_single.py, a CPU-only analysis script with no GPU/
    model in the loop. `probe_ids`/`all_members` DO depend on true single-token-ness
    and should not be trusted from this stub (encode() just splits on whitespace, so
    a multi-word animal name reports as multiple "tokens" -- a reasonable proxy, but
    not the real tokenizer's answer)."""
    def encode(self, s, add_special_tokens=False):
        return s.split()


def load_troyer_norms(csv_path, tok, drop_full_columns=False, verbose=True):
    """
    Returns dict with:
      categories : {cat: set(surface_word)}         full membership
      word2cats  : {surface_word: set(cat)}         reverse index
      probe_ids  : {surface_word: token_id}         single-token members only
      all_members: set(surface_word)
      dropped    : [cat, ...]                        columns dropped as suspect

    drop_full_columns defaults to OFF: it used to drop any column whose
    unique-member count >= row count, on the theory that meant "export
    artifact" -- but that heuristic can't tell a genuinely complete, diverse
    category (e.g. `water`, 241 distinct aquatic species across 241 rows)
    from an actual artifact (the same junk value repeated down every row).
    It only ever flagged `water` on this dataset, which is real data, not an
    artifact -- so it's off by default. Pass True only if you've separately
    confirmed a specific column in your CSV is bogus.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    n_rows = len(rows)
    categories = {h: set() for h in header}
    col_value_counts = {h: 0 for h in header}   # non-blank cell count per column
    for row in rows:
        for i, cell in enumerate(row):
            w = _norm(cell)
            if w:
                categories[header[i]].add(w)
                col_value_counts[header[i]] += 1

    # If ever re-enabled: flag columns where almost every filled cell repeats
    # the SAME value (the actual signature of an export artifact), not just
    # "unique count >= row count" (which a real, complete category also has).
    dropped = []
    if drop_full_columns:
        for h in list(categories):
            n_filled = col_value_counts[h]
            n_unique = len(categories[h])
            if n_filled > 0 and n_unique <= max(1, 0.05 * n_filled):
                dropped.append(h)
                del categories[h]

    # Reverse index over the surviving categories.
    word2cats = {}
    for cat, members in categories.items():
        for w in members:
            word2cats.setdefault(w, set()).add(cat)
    all_members = set(word2cats)

    # Single-token probe set: keep animals that are ONE leading-space token.
    probe_ids = {}
    for w in all_members:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            probe_ids[w] = ids[0]

    if verbose:
        print(f"[norms] {len(categories)} categories kept, "
              f"{len(dropped)} dropped {dropped}", flush=True)
        print(f"[norms] {len(all_members)} total animals, "
              f"{len(probe_ids)} single-token probes", flush=True)
        # coverage per category among probes
        for cat in sorted(categories):
            n_probe = sum(1 for w in categories[cat] if w in probe_ids)
            print(f"[norms]   {cat:20s} total={len(categories[cat]):3d} "
                  f"single-token={n_probe:3d}", flush=True)

    return {
        "categories": categories,
        "word2cats": word2cats,
        "probe_ids": probe_ids,
        "all_members": all_members,
        "dropped": dropped,
    }


def _naive_plural(word):
    """Simple English pluralization -- ends in s/x/z/ch/sh -> +es; consonant+y ->
    -y+ies; else +s. Good enough for common food nouns (see load_food_norms's
    add_plurals); not a general-purpose lemmatizer (irregular plurals like "leaf"/
    "leaves" or "child"/"children" aren't handled, and aren't common enough in this
    vocabulary to be worth a dependency for)."""
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def load_food_norms(csv_path, tok, merge_categories=None, add_plurals=True, verbose=True):
    """
    Loader for the SNAFU-style food-fluency norms (foods_snafu_scheme.csv), for the
    food-fluency generalization variant of exp6. Structurally DIFFERENT from
    load_troyer_norms's wide-format animal CSV: this is a plain LONG-format CSV,
    "category,item" one pair per row, with no header row naming the two columns (the
    file opens with two free-text comment/attribution lines instead, which are
    SKIPPED as non-data, not parsed) and no underscore-joined-multiword convention at
    all -- every item is written as ONE run-together, unspaced string, whether it is
    linguistically a single word (e.g. "banana") or a compressed multi-word dish name
    (e.g. "chickenfriedrice", "hamandcheesesandwich"). There is no `_norm`-style
    space-insertion step here (nothing to insert); items are used exactly as given,
    just lowercased/stripped.

    Returns the SAME dict shape as load_troyer_norms (categories, word2cats,
    probe_ids, all_members, dropped) so it's a drop-in for every existing consumer
    (is_switch, build_tracked_sequence(_full), within_patch_members,
    word_detection_ids, build_leading_token_groups, etc.) -- `dropped` is always []
    here (load_troyer_norms's drop_full_columns heuristic doesn't apply; this format
    has no "columns" to drop).

    merge_categories: {kept_name: [alias, ...]} -- members of each alias category are
    folded into kept_name and the alias key is removed. Defaults to merging the source
    file's "Breakfest" into "Breakfast" (the source CSV has both as separate
    categories -- 7 items under "Breakfast": cereal/toast/waffle/frenchtoast/
    hashbrowns/syrup/boildedegg, 4 under "Breakfest": omelette/pancake/quiche/
    scrambledeggs -- almost certainly one category split by a spelling typo in data
    collection, confirmed as a merge-worthy duplicate rather than a deliberate
    distinction). Pass merge_categories={} to keep the file exactly as delivered
    instead (e.g. for exact reproducibility against a citation of the raw norms).

    add_plurals (default True): the source file lists items in SINGULAR form only --
    checked systematically, 646/647 items have no plural counterpart in the
    vocabulary at all. A model naming food items very naturally pluralizes ("tacos",
    "noodles", "eggs", "onions", "fries") and every one of those was previously
    invisible to build_tracked_sequence_full: not a rare edge case, a majority-of-
    real-output miss that inflated zero-emission rates (confirmed as high as 22% in
    one model's UNSTEERED baseline) and likely also inflated the measured switch rate
    by turning what should have been ordinary same-category continuations into gaps
    or missed transitions. When True, each singular item gets a naive English plural
    alias (see _naive_plural) registered under the SAME category set, UNLESS that
    exact string already denotes a different, existing item in the file (in which
    case the file's own entry is left untouched, never overwritten -- e.g. the file
    already lists "vegetables"/"fruits" as their own items, distinct from a singular
    "vegetable"/"fruit" that isn't in the file at all).

    KNOWN TASK-DESIGN INTERACTION, not something this loader can fully fix: ~500 of
    647 items in this file are multi-word-compressed-to-one-string dish names (e.g.
    "generaltsoschicken"), which a model instructed to name ONE-WORD food items is
    highly unlikely to ever spontaneously produce verbatim (it would naturally say
    "general tso's chicken" with spaces). Categories dominated by such compounds
    (Asianfood/Europeanfood/Fastfood/Italianfood/Middleeasternfood/Southernfood/
    Spanishfood) should be expected to have much sparser real coverage than
    categories dominated by genuine single words (Fruit/Vegetable/Meat/Seafood/Nut/
    Dairy/Candy/Drink/Seasoning/Grain/Snack) -- a property of the task/norms-file
    combination that add_plurals does not address (pluralizing a compound dish name
    doesn't make it any more one-word).
    """
    if merge_categories is None:
        merge_categories = {"Breakfast": ["Breakfest"]}

    categories = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    n_skipped = 0
    for row in rows:
        if not row or row[0].strip().startswith("#") or len(row) != 2:
            n_skipped += 1
            continue
        cat, item = row[0].strip(), row[1].strip().lower()
        if not cat or not item:
            n_skipped += 1
            continue
        categories.setdefault(cat, set()).add(item)

    for kept, aliases in merge_categories.items():
        for alias in aliases:
            if alias in categories:
                categories.setdefault(kept, set()).update(categories.pop(alias))

    if add_plurals:
        all_items_before = {w for members in categories.values() for w in members}
        n_added = 0
        for cat in list(categories.keys()):
            for w in list(categories[cat]):
                p = _naive_plural(w)
                if p not in all_items_before:
                    categories[cat].add(p)
                    n_added += 1
        if verbose:
            print(f"[food_norms] added {n_added} naive-plural aliases "
                  f"(e.g. 'taco'->'tacos') -- see load_food_norms's add_plurals docstring", flush=True)

    word2cats = {}
    for cat, members in categories.items():
        for w in members:
            word2cats.setdefault(w, set()).add(cat)
    all_members = set(word2cats)

    probe_ids = {}
    for w in all_members:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            probe_ids[w] = ids[0]

    if verbose:
        print(f"[food_norms] {len(categories)} categories, {len(all_members)} total items, "
              f"{len(probe_ids)} single-token probes ({n_skipped} non-data rows skipped)", flush=True)
        for cat in sorted(categories):
            n_probe = sum(1 for w in categories[cat] if w in probe_ids)
            print(f"[food_norms]   {cat:20s} total={len(categories[cat]):3d} "
                  f"single-token={n_probe:3d}", flush=True)

    return {
        "categories": categories,
        "word2cats": word2cats,
        "probe_ids": probe_ids,
        "all_members": all_members,
        "dropped": [],
    }


def build_word_of_id(tok, probe_ids):
    """Reverse lookup {token_id: word} for detecting which probe word was
    emitted in generated text. Includes BOTH the lowercase form's token id
    (the one probe_ids/J-lens directions are built from) AND, if distinct
    and itself single-token, the Title-Case form's token id.

    Free generation sometimes capitalizes every list item (e.g. "Lion,
    Tiger, Bear, ..." rather than "lion, tiger, bear, ..."), which
    tokenizes to a DIFFERENT id than the lowercase form -- silently
    invisible to token-level emission detection built only from probe_ids.
    Observed empirically to zero out ALL detected transitions/emissions in
    ~28% of sampled rollouts (every layer, for the affected seeds) before
    this fix -- not random noise, a systematic loss of a large fraction of
    otherwise-valid data.
    """
    word_of_id = {}
    for w, tid in probe_ids.items():
        word_of_id[tid] = w
        cap_ids = tok.encode(" " + w.capitalize(), add_special_tokens=False)
        if len(cap_ids) == 1 and cap_ids[0] != tid:
            word_of_id[cap_ids[0]] = w
    return word_of_id


def is_switch(word_a, word_b, word2cats):
    """True if consecutive emitted animals share NO category (a patch switch)."""
    ca = word2cats.get(word_a, set())
    cb = word2cats.get(word_b, set())
    return len(ca & cb) == 0


def within_patch_members(word, word2cats, categories):
    """All animals sharing >=1 category with `word` (its patch neighborhood)."""
    cats = word2cats.get(word, set())
    out = set()
    for c in cats:
        out |= categories[c]
    out.discard(word)
    return out


def split_generated_list(text):
    """Comma-separated fluency output -> cleaned lowercase items. A multiword
    norms member (e.g. "whale_shark" -> "whale shark") survives as ONE item
    here, since it has no internal comma -- this is what makes matching it
    against `all_members` correct, as opposed to per-token matching, which
    can only ever see single-token members."""
    items = [x.strip().lower().rstrip(".") for x in text.split(",")]
    return [x for x in items if x]


def build_tracked_sequence(gen_text, gen_ids, prompt_len, word_of_id):
    """Single-token-probe emissions, in order, each tagged with whether
    something else (a multiword norms member like "whale shark", or any
    other unrecognized item) was produced immediately before it in the
    comma-separated list -- i.e. whether it's TRULY adjacent to the previous
    probe emission, or only *appears* adjacent because token-level matching
    silently skips anything that isn't a single-token probe.

    Without this, exp2's switch/stay logic would treat two probe emissions
    with an intervening (but invisible-to-token-matching) animal as a direct
    transition, corrupting supply/run_len/switch counts around any gap.

    Returns [(abs_token_pos, word, gap_before), ...]. `gap_before` is True
    when this emission is NOT the item immediately following the previous
    tracked emission in the comma list -- callers should skip transitions
    into a gapped emission rather than scoring them.
    """
    items = split_generated_list(gen_text)
    probe_words = set(word_of_id.values())    # word_of_id is keyed by token id, not word

    probe_hits = [(prompt_len + o, word_of_id[t]) for o, t in enumerate(gen_ids) if t in word_of_id]
    probe_item_idx = [i for i, it in enumerate(items) if it in probe_words]

    # Both lists are built by a single left-to-right scan over the same
    # underlying content (token stream vs. comma-split text), so -- barring a
    # probe word appearing as a spurious substring of some other item, which
    # whole-item matching here rules out -- they line up 1:1 in order.
    n = min(len(probe_hits), len(probe_item_idx))
    out = []
    prev_item_idx = None
    for k in range(n):
        abs_pos, w = probe_hits[k]
        item_idx = probe_item_idx[k]
        gap = prev_item_idx is not None and item_idx != prev_item_idx + 1
        out.append((abs_pos, w, gap))
        prev_item_idx = item_idx
    return out


def build_tracked_sequence_ext(gen_text, gen_ids, prompt_len, word_of_id, all_members):
    """Same left-to-right alignment as build_tracked_sequence (first 3 return-tuple
    elements are identical), but ALSO returns, per single-token tracked event, the
    set of ALL comma-list items -- single- OR multi-token, matched against
    `all_members` -- that appeared strictly BEFORE it in the raw generated list. A
    multi-token norms member (e.g. "whale shark") is invisible to the single-token
    tracked sequence itself (build_tracked_sequence can't detect it, and it never
    becomes a `word`/`next` in any event), but a caller building an EXTENDED
    same-category "supply" measure (see exp2_ext_supply_single.py) needs to know
    whether such a word has already been said, even though it isn't itself trackable.
    Kept as a SEPARATE function, not a signature change to build_tracked_sequence --
    that function is relied on by exp1/exp2/exp5/exp6 with its existing 3-tuple
    contract, and this extra computation (an O(n) slice per event) is only needed by
    the extended-supply pipeline.
    Returns [(abs_token_pos, word, gap_before, full_emitted_before), ...] where
    full_emitted_before is a frozenset of item strings."""
    items = split_generated_list(gen_text)
    probe_words = set(word_of_id.values())

    probe_hits = [(prompt_len + o, word_of_id[t]) for o, t in enumerate(gen_ids) if t in word_of_id]
    probe_item_idx = [i for i, it in enumerate(items) if it in probe_words]

    n = min(len(probe_hits), len(probe_item_idx))
    out = []
    prev_item_idx = None
    for k in range(n):
        abs_pos, w = probe_hits[k]
        item_idx = probe_item_idx[k]
        gap = prev_item_idx is not None and item_idx != prev_item_idx + 1
        full_emitted_before = frozenset(it for it in items[:item_idx] if it in all_members)
        out.append((abs_pos, w, gap, full_emitted_before))
        prev_item_idx = item_idx
    return out


def word_detection_ids(word, tok, cache=None):
    """The token id(s) that mark word BEING STARTED in a raw generated token
    stream -- for a single-token word this is its own (only) token id; for a
    multi-token word it's the FIRST token of its encoding (the standard
    mech-interp leading-token heuristic: the model can't say "hippopotamus"
    without first emitting whatever token begins it -- see
    build_leading_token_groups). Both the lowercase and Title-Case surface
    forms are checked and unioned: free generation sometimes capitalizes
    every list item (e.g. "Lion, Tiger, Bear, ..."), which tokenizes to a
    DIFFERENT leading id than the lowercase form -- see build_word_of_id,
    which found this silently zeroed out ALL single-token detections in
    ~28% of sampled rollouts before being fixed there; the identical risk
    applies to multi-token leading-token detection and is fixed here the
    same way, for both single- and multi-token words alike (one rule
    covers both, since "first token of the encoding" already reduces to
    "the only token" when the word IS single-token).
    Returns a 1-2 element frozenset. `cache`, if given a dict, memoizes
    per-word lookups (tok.encode is not free and this runs once per tracked
    item per generated sequence)."""
    if cache is not None and word in cache:
        return cache[word]
    ids = set()
    for form in (word, word.capitalize()):
        enc = tok.encode(" " + form, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    result = frozenset(ids)
    if cache is not None:
        cache[word] = result
    return result


def build_tracked_sequence_full(gen_text, gen_ids, prompt_len, all_members, tok, cache=None):
    """Full-coverage switch/stay tracked sequence: unlike build_tracked_sequence
    (single-token PROBE emissions only), this tracks every RECOGNIZED category
    member in the generated comma list -- single- or multi-token alike -- as a
    first-class event with its own position. A multi-token emission (e.g. "whale
    shark") is no longer an invisible gap that silently corrupts the switch/stay
    label of the transitions immediately around it, and can itself be the origin OR
    destination of a scored transition. This is the OUTCOME-side counterpart to
    build_tracked_sequence_ext / exp2_ext_supply's supply_ext (which only extends
    the PREDICTOR): that leaves "what counts as a switch" untouched; this changes
    it, per the underlying concern that single-token-only observability biases not
    just the supply count but which transitions get scored as switch/stay events at
    all (exp5/exp6's selection-bias problem, not just exp2's biased-count problem).

    Detection position: for a single-token word this is that token's own position
    (identical to build_tracked_sequence). For a multi-token word there's no single
    position meaning "the whole word was said" -- its LEADING token's position is
    used instead (see word_detection_ids), the same proxy build_leading_token_groups
    already uses for the predictor side. Consequently the activation associated with
    a multi-token tracked event reflects only its first token, not the full word --
    an inherent limitation of any token-level probe of a multi-token concept, not
    something this function can fix.

    Ambiguity: if two candidate words share a leading token (e.g. both start with
    " sea"), a naive scan of gen_ids can't tell which was meant by id alone --
    resolved here by using the independently-parsed comma-list TEXT
    (split_generated_list) to know which SPECIFIC word is expected next, and
    searching gen_ids for just THAT word's own detection ids, starting after the
    previous tracked item's match position. This relies on item order in the text
    matching token order in generation (true by construction: the text IS the
    decoded token stream) and walks strictly forward, so it cannot mis-order items;
    an item whose detection token isn't found downstream (rare -- e.g. a tokenizer
    edge case) is SKIPPED (excluded, not mis-positioned), which surfaces as
    `gap_before=True` on whatever tracked item follows it.

    Returns [(abs_token_pos, word, gap_before), ...] -- the SAME 3-tuple shape as
    build_tracked_sequence, so is_switch(word_a, word_b, word2cats) and every
    existing consumer of that shape work unmodified; `word` may now be a
    multi-token surface form where it previously never could be. `gap_before` is
    True only when an item between two tracked items was NOT itself a recognized
    all_members word (e.g. a hallucinated non-animal) or its detection token
    couldn't be located -- with full coverage this should be rare: 0/327
    single-token-only "gaps" were anything other than an ordinary recognized
    multi-token animal in a 600-rollout spot-check against real Gemma-9B exp2 data.
    `cache`, if given a dict, memoizes per-word leading-id lookups across calls
    (recommended -- pass one dict per process and reuse it across seeds)."""
    if cache is None:
        cache = {}
    items = split_generated_list(gen_text)
    tracked_items = [(i, it) for i, it in enumerate(items) if it in all_members]

    out = []
    prev_item_idx = None
    search_from = 0
    for item_idx, word in tracked_items:
        want = word_detection_ids(word, tok, cache)
        pos = None
        for o in range(search_from, len(gen_ids)):
            if gen_ids[o] in want:
                pos = o
                break
        if pos is None:
            continue    # detection token not found downstream -- skip, don't guess
        gap = prev_item_idx is not None and item_idx != prev_item_idx + 1
        out.append((prompt_len + pos, word, gap))
        prev_item_idx = item_idx
        search_from = pos + 1
    return out


def build_leading_token_groups(word2cats, probe_ids, tok):
    """For every animal NOT in probe_ids (i.e. multi-token, invisible to the ordinary
    single-token J-lens/probe pipeline), find its leading token id(s) via
    word_detection_ids -- the standard mech-interp heuristic for probing a
    multi-token concept via the token that commits to starting to say it. A word
    contributes ONE GROUP ENTRY PER DISTINCT leading id it has (lowercase and, if
    different, Title-Case -- see word_detection_ids), so it can appear in up to 2
    groups; callers that need "is this word's supply visible" must therefore check
    across ALL of a word's leading-id groups, not just one (see build_patch_pool_ext,
    which already does this correctly).
    Multiple multi-token words can share the same leading token -- grouped together,
    since the activation machinery can only ever see "this leading token is active,"
    not which specific word it will complete into (a real, inherent limitation of
    this heuristic -- documented at the call site in exp2_ext_supply_single.py, not
    hidden). Leading tokens that COLLIDE with an existing single-token probe's own
    token id are EXCLUDED: that matrix column already represents a specific whole
    word, and conflating it with a different word's leading-token proxy would be
    incoherent.
    Returns {leading_token_id: [word, ...]}, sorted-word-order per group for
    determinism."""
    single_token_ids = set(probe_ids.values())
    groups = {}
    for w in word2cats:
        if w in probe_ids:
            continue
        seen = set()
        for lead in word_detection_ids(w, tok):
            if lead in single_token_ids or lead in seen:
                continue
            seen.add(lead)
            groups.setdefault(lead, []).append(w)
    return {lead: sorted(words) for lead, words in groups.items()}


def build_patch_pool_ext(word2cats, categories, index_words, cidx, leading_groups):
    """Extended same-category neighborhood per word in `index_words` (originally
    just the single-token probe words; now also called with ALL recognized words --
    see build_tracked_sequence_full -- since a full-coverage tracked sequence can
    make a multi-token word itself the ORIGIN of a transition, not just a candidate
    neighbor), split into the two kinds of candidate the extended activation matrix
    can represent:
      * "single": indices into the ORIGINAL single-token candidate array (cidx) --
        identical to the existing patch_pool.
      * "multi": (leading_group_index, member_word) pairs -- leading_group_index is
        0-based position within `sorted(leading_groups)` (the caller offsets this by
        the single-token block size C to get a column in the COMBINED activation
        matrix); member_word is kept per-pair (not deduplicated by group) so the
        caller can still check each specific word's own emitted status even when
        several same-category words share one leading-token group. A word with TWO
        leading-id groups (lowercase + Title-Case, see build_leading_token_groups)
        contributes ONE "multi" entry per group it belongs to -- callers counting
        supply must dedupe by member_word (either group being active means that
        word's supply is visible), not just count pairs; see
        exp2_ext_supply_single.py's supply_multi computation.
    Returns (pool_ext, leading_ids_sorted) where leading_ids_sorted is the token-id
    order the caller must build the extended candidate/activation columns in."""
    word_to_leadings = {}
    for lead, words in leading_groups.items():
        for w in words:
            word_to_leadings.setdefault(w, []).append(lead)
    leading_ids_sorted = sorted(leading_groups.keys())
    group_idx = {lead: i for i, lead in enumerate(leading_ids_sorted)}

    pool_ext = {}
    for w in index_words:
        nbrs = within_patch_members(w, word2cats, categories)
        single = [cidx[m] for m in nbrs if m in cidx]
        multi = [(group_idx[lead], m) for m in nbrs if m in word_to_leadings
                 for lead in word_to_leadings[m]]
        pool_ext[w] = {"single": single, "multi": multi}
    return pool_ext, leading_ids_sorted


if __name__ == "__main__":
    # quick self-test with the Gemma tokenizer if available, else a stub
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "extenden_nundin_norms.csv"
    try:
        from transformers import AutoTokenizer
        mp = os.environ.get("MODEL_PATH", "google/gemma-2-2b")
        tok = AutoTokenizer.from_pretrained(mp)
    except Exception as e:
        print("no HF tokenizer, using whitespace stub:", e)
        class Stub:
            def encode(self, s, add_special_tokens=False):
                return s.split()  # single token iff single word
        tok = Stub()
    info = load_troyer_norms(csv_path, tok)
    print("\nexample word2cats['koala'] =", info["word2cats"].get("koala"))
