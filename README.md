# LogXform

**A retrieval signal that matches *how* someone reasons, not *what* they reason about.**

Sentence embeddings are dominated by topic. Two texts about medicine sit closer than a medical text and a philosophical one, no matter how similar the *reasoning* underneath. That makes ordinary semantic search blind to a real thing: the fact that you debugged a circuit the same way you ran a vendor negotiation — same elimination strategy, different domain.

LogXform is one operation that exposes that hidden axis. It applies `log(|x| + ε)` to each embedding dimension before taking variance. The log compresses the few large dimensions that carry topic and stretches the many small ones that carry structure, so the resulting fingerprint describes *structural disagreement* across a cluster rather than *topical agreement*. Match on that fingerprint and you retrieve cross-domain structural resonance instead of more-of-the-same-topic.

That's the whole idea. The rest of this repo is the math, a runnable implementation, the experimental record, and — importantly — an honest account of what is demonstrated versus what is still hypothesis.

---

## What's actually shown, and what isn't

This matters more than the pitch, so it's near the top.

**Shown (measured, on a real 188-cluster / 16,736-pair test set):**
- The log transform inverts the scale hierarchy of MiniLM embeddings, which are known to be anisotropic — a few "rogue" dimensions dominate cosine similarity (Timkey & van Schijndel, 2021). The fingerprint is effectively a magnitude-blind, per-dimension variance profile.
- Against ten other candidate transforms tested head-to-head, log-variance was the only one that produced a *low* false-positive rate (13.2%) **and** a non-trivial set of cross-topic matches (23). Every alternative either matched almost everything (raw variance 48.9%, rank 67.5%) or killed all signal (squared 1.2%, top-k 0%). The full table is in the [spec](spec/logxform-spec.md#method-comparison).
- The discovery process disproved its own first hypothesis: PCA was predicted to help if the problem were high-dimensional geometry; it made things *worse* at every dimension count, which is what redirected the work toward the semantic-dominance explanation that turned out to be right.

**Not yet shown (these are the gates between "working system" and "publishable result"):**
- That the 23 matches mean anything. They were confirmed by an LLM, which is itself embedding-based and may be recognizing surviving semantic traces, not structure. **Until the [null-baseline experiment](experiments/README.md#1-null-baseline-the-publication-gate) runs, the 23/23 confirmation rate is uncalibrated.** This is the single most important missing control.
- That the signal is *cognitive* rather than *stylometric*. Epistemic posture and writing register are confounded in text, and the residual dimensions are exactly where syntax and register live. LogXform may be a very good *stylometric* fingerprint wearing a cognitive name. The [style/structure disentanglement experiment](experiments/README.md#2-stylestructure-disentanglement) decides which.
- That it's substrate-agnostic. The result is MiniLM-specific until it replicates under a second embedding model.

If you take nothing else from this README: **the math is real and the empirical process was honest, but the central interpretive claim currently rests on validation that hasn't been calibrated against a null.** That experiment is cheap, specified, and runnable on data that already exists. It's the next thing, not a someday thing.

---

## Try it

```bash
pip install sentence-transformers numpy
python logxform.py "first block of reasoning" "second block of reasoning"
```

Or paste two blocks interactively:

```bash
python logxform.py
```

It returns three numbers:
- **shape similarity** — structural (how it's reasoned)
- **semantic similarity** — topical (what it's about)
- **delta** — shape minus semantic; high delta is the target signal (same shape, different topic)

Add `--analyze` for a content-side breakdown — the most central ("thesis") sentence of each block, each block's internal topical spread, and the most-aligned and most-divergent sentence pairs across the two:

```bash
python logxform.py --analyze "first block" "second block"
```

The two readouts are kept deliberately separate: the comparison scores the structural axis, the analysis examines the semantic axis. That separation *is* the method's claim — that structure and content are different things — so the tool shows both rather than blending them.

The standalone tool clusters over the *sentences* of each paste. The production system clusters over *graph nodes* in a knowledge graph. The math is identical; the unit of clustering differs, which is why the tool prints raw scores and treats the match/no-match line as a reading rather than a verdict.

---

## The method in five lines

For a cluster of embeddings `E` (each a 384-dim unit vector):

```
1.  T = ln(|E| + 1e-6)          # log-transform every dimension
2.  v = var(T, axis=cluster)    # per-dimension variance across the cluster
3.  fingerprint = v / ‖v‖       # L2-normalize
4.  shape_sim    = cos(fp_a, fp_b)              # structural axis
5.  semantic_sim = cos(mean(E_a), mean(E_b))    # topical axis
```

A match (in production) requires `shape_sim > 0.65` **and** `semantic_sim < 0.45` — structurally alike, topically apart. The full algorithm, including graph expansion and the production thresholds, is in [`spec/logxform-spec.md`](spec/logxform-spec.md).

---

## Repository

```
logxform.py                two-paste reference implementation (start here)
THEORY.md                  demonstrated vs. speculated vs. the tests — read this to evaluate
spec/logxform-spec.md      the exact math, extracted from production code
FRAMING.md                 claims to make, claims to retire — the discipline
experiments/README.md      full protocols for the controls
```

Read order if you're evaluating it: **THEORY → spec → experiments → FRAMING**. THEORY.md is the one that keeps demonstrated, hypothesized, and unproven cleanly apart — it's where you find out exactly what the evidence does and doesn't support before reading the math.

---

## Provenance

LogXform is the structural-fingerprinting layer of **The Orchard**, a persistent knowledge-graph architecture (USPTO Provisional #63/979,094). It runs in production on-device, in SQLite, on a phone. This repository extracts the method as a standalone, substrate-independent component so it can be evaluated on its own terms.

Method, code, and experimental record by Chris (cedenburn-ai). The method was discovered empirically — eleven transforms tested and eliminated — and given a mathematical explanation afterward, not derived from theory. That order is in the spec, kept because the elimination record is part of the evidence.

License: see [LICENSE](LICENSE).
