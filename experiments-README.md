# Experiments

These are the controls that move LogXform from "working system" to "validated result." They are ordered by leverage. The first one is the gate — nothing should be claimed publicly as cognitive-structure detection until it runs.

Each is specified so someone other than the author could run it and get a number that either confirms or collapses a specific claim. Where data already exists, that's noted.

---

## 1. Null baseline (the publication gate)

**The claim it tests:** that LogXform's 23 cross-topic matches are real structural isomorphism, not LLM-confirmation artifact.

**Why it's the gate:** the 23 matches were each confirmed by an LLM judge. But an LLM asked "do these two reasoning traces share a methodology?" will find isomorphism between almost *any* two structured human reasonings, because almost anything can be analogized at some level of abstraction. A 23/23 confirmation rate is meaningless without knowing the judge's baseline confirmation rate on pairs that should *not* match.

**Protocol:**
1. Construct a null set: random cross-topic pairs that **pass the semantic ceiling** (`semantic_sim < 0.45`, so they're genuinely different topics) but **fail the shape threshold** (`shape_sim < 0.65`, so LogXform says they're *not* structurally alike).
2. Present each null pair to the **same LLM judge, with the exact same prompt** used for the 23, blind — the judge must not know which pairs are LogXform matches and which are nulls. Interleave them.
3. Record the judge's confirmation rate on the null set.

**Reading the result:**
- Null confirmation ≈ 70% → the judge confirms anything; the 23/23 tells you almost nothing; the result is not yet evidence.
- Null confirmation ≈ 5% → LogXform is selecting pairs the judge treats very differently from random cross-topic pairs; the 23 become genuinely strong.
- The publishable number is the **separation** between the match-set confirmation rate and the null-set confirmation rate, not 23/23 in isolation.

**Status:** not run. This is the experiment to run before claiming anything.

---

## 2. Style/structure disentanglement

**The claim it tests:** that the fingerprint tracks *reasoning* (cognitive), not *register* (stylometric).

**Why it matters:** if LogXform follows writing style rather than reasoning method, it's still useful, but it's a stylometric tool and should be named and positioned as one — against the authorship-attribution literature, not the cognitive one.

**Protocol — a 2×2 cross design:**
1. Take a reasoning trace R in register S.
2. Generate R in a *different* register S′ (LLM paraphrase preserving the reasoning, changing the prose).
3. Generate *different* reasoning R′ in the *original* register S (matched style, different method).
4. Fingerprint all four. Ask: do fingerprints cluster by **reasoning** (R, R-in-S′ together) or by **register** (R, R′ together)?

**Reading the result:**
- Fingerprints follow the reasoning across register change → the cognitive interpretation holds; "cognitive fingerprint" earns its name.
- Fingerprints follow the register → it's stylometry; rename and reposition. Still a working tool, different claim.

**Status:** not run. Settles the single most important naming question (see FRAMING.md).

---

## 3. ε sensitivity sweep

**The claim it tests:** that the result is not an artifact of the `1e-6` log floor.

**Protocol:** recompute the 23 matches (and the FP rate on the full pair set) across ε ∈ {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8}. Report match stability and FP-rate stability as a function of ε.

**Reading:** matches survive across several orders of magnitude → ε is a robust floor, not a tuned knob. Matches appear/vanish with ε → the result is partly numerical-noise-driven and the floor needs principled selection.

**Status:** not run. Cheap. Existing data.

---

## 4. Cross-embedding replication

**The claim it tests:** substrate-agnosticism.

**Protocol:** recompute fingerprints for the matched pairs under a second embedding model (e.g. GTE-small, or any non-MiniLM sentence encoder). Check whether the same pairs match.

**Reading:** same pairs match under a different encoder → the signal is in the reasoning structure, not in MiniLM-specific artifacts; "substrate-agnostic" earns its place. Matches don't transfer → the method is MiniLM-specific and should be scoped that way.

**Status:** not run. One run rules out a whole class of "it's just a MiniLM quirk" objections.

---

## 5. Cluster-size confound check

**The claim it tests:** that shape similarity isn't a proxy for cluster-size similarity.

**Protocol:** across the matched pairs, plot shape similarity against the absolute difference in node count. Compute the correlation.

**Reading:** near-zero correlation → cluster size is not driving matches. Strong correlation → small clusters match small clusters for sampling-noise reasons, and the result needs size-controlled re-analysis.

**Status:** not run. One plot. Existing data.

---

## Running notes

- Experiments 3, 4, and 5 run on data that already exists and are mechanical — they could be done in an afternoon and they harden the result against the cheapest objections.
- Experiments 1 and 2 require new judge runs but are the ones that decide whether the central claim survives.
- The honest closing position, until #1 and #2 run: *LogXform is a structural fingerprint with a low measured false-positive rate, whose interpretation as cognitive (rather than stylometric) and whose validation against a null are open and specified.* That sentence is defensible to a hostile reviewer today. The experiments are what let you delete the hedges.
