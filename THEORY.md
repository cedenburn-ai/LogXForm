# Theory, Speculation, and the Tests That Separate Them

This file exists so the line between *demonstrated*, *hypothesized*, and *unproven-but-testable* is never blurred. Everything about LogXform falls into one of three columns. The method's credibility depends on keeping them apart — a reader should be able to see exactly where the evidence stops and the interpretation begins.

Read this before the spec if you are evaluating the claim. Read the spec for the math.

---

## Column 1 — DEMONSTRATED (measured, reproducible from the record)

These are backed by the 188-cluster / 16,736-pair experiment, by code inspection, or by published prior art.

| Claim | Evidence |
|---|---|
| `log(\|x\| + ε)` inverts the scale hierarchy of MiniLM embeddings. | Math is explicit; MiniLM anisotropy is documented (Timkey & van Schijndel 2021). The fingerprint is a magnitude-blind per-dimension variance profile. |
| Among 11 transforms tested head-to-head, log-variance uniquely produced low FP (13.2%) *and* a non-zero cross-topic match set (23). | The full comparison table (raw variance 48.9%, rank 67.5%, squared 1.2%, top-k 0%). Every alternative failed in an interpretable direction. |
| The high-dimensional-geometry explanation is wrong. | PCA was predicted to help if convergence were a curse-of-dimensionality effect. It made matching *worse* at every dimension count. Hypothesis tested and disproved on the record. |
| Internal consistency holds. | XOR fails by parity (predicted); squared collapses the exact residual signal the theory says it should (predicted). |

These claims survive a hostile reviewer today. State them plainly, without hedges.

---

## Column 2 — SPECULATED (the interpretation; plausible, not yet established)

These are where the *meaning* of the demonstrated signal is asserted. Each is a hypothesis, and the repo's discipline is to name them as such until Column 3 closes them.

| Speculative claim | Why it's not yet demonstrated |
|---|---|
| The fingerprint captures **cognitive posture** ("how someone reasons"). | "Epistemic posture" and "writing register" are confounded in text; the residual dimensions where the fingerprint lives are exactly where style, syntax, and lexical statistics live. It may be a **stylometric** fingerprint wearing a cognitive name. (Still useful — differently named.) |
| The 23 matches are **genuine structural isomorphism**. | They were confirmed by an LLM judge, which is embedding-based and prone to finding isomorphism between almost any two structured reasonings. The true positives are defined by the judge being tested — **circular until a null calibrates it**. |
| The method is **substrate-agnostic**. | Result is MiniLM-specific until it replicates under a second embedding model. |
| The ε = 1e-6 floor is **principled**. | `ln(1e-6) ≈ -13.8`, so near-zero dimensions can dominate variance from numerical noise. The floor is a tuned parameter until a sweep shows the matches are stable across it. |
| The retrieval signal is independent of **cluster size**. | Variance from 3 embeddings vs. 30 has very different sampling noise; unconfirmed until correlation is checked. |
| The **Pribram / holonomic** analogy is a literature position. | It is an evocative intuition only; holonomic theory makes Fourier-domain dendritic claims LogXform does not instantiate. Hold loosely or drop. |

The entire value of Column 2 becoming Column 1 rests on the experiments below. Nothing here should be stated publicly as fact.

---

## Column 3 — THE TESTS (in order of yield)

This is the attack surface, ordered so the highest-leverage experiment comes first. Run them in this order; the first two decide whether the central claim survives, and items 3–5 are cheap and run on existing data.

**(1) Null baseline — the make-or-break.**
Same judge, same prompt, blind, on random cross-topic pairs that *pass* the semantic ceiling but *fail* the shape threshold. The judge's confirmation rate on these nulls is the number that makes or breaks the 23/23. ~70% on nulls → the result is noise. ~5% → it's strong. *Closes: "genuine isomorphism."*

**(2) Style/structure disentanglement.**
Same reasoning paraphrased into a different register, vs. different reasoning in matched register. See which the fingerprint follows — the reasoning (cognitive) or the register (stylometric). *Closes: "cognitive vs. stylometric."*

**(3) ε sensitivity sweep.**
Recompute matches and FP rate across ε ∈ [1e-3, 1e-8]. Matches should survive the full range if the floor is principled. *Closes: "ε is principled."* (Cheap, existing data.)

**(4) Cluster-size confound.**
Correlate shape similarity with node-count difference across matched pairs. Near-zero correlation clears it. *Closes: "size-independent."* (One plot, existing data.)

**(5) Cross-embedding replication.**
Recompute fingerprints under a second embedding model (e.g. GTE-small). Surviving matches rule out MiniLM-specific artifacts and are the strongest available test of substrate-agnosticism. *Closes: "substrate-agnostic."*

Full protocols for each are in [`experiments/README.md`](experiments/README.md).

---

## The honest one-sentence status

Until tests (1) and (2) run:

> LogXform is a **structural fingerprint** with a low measured false-positive rate (13.2% over 16,736 pairs), whose interpretation as *cognitive* rather than *stylometric*, and whose validation against a null, are open and specified.

That sentence is defensible to a hostile reviewer right now. The experiments are what let you delete the word "structural" and earn the word "cognitive" — or keep "structural," rename, and still have a working tool. Either outcome is a result. Neither requires overclaiming today.
