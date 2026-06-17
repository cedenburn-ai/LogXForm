# Framing Discipline

The work is strong enough that overclaiming is the main risk to it. A reviewer who catches one inflated claim discounts the rest, including the parts that are solid. So this file states, on the record, which claims the evidence currently supports and which are hypotheses wearing finding's clothes.

This is not modesty. It's that the calibrated version is more persuasive than the inflated one, and the inflated one is fragile.

---

## Claims the evidence supports — make these

- **The transform does what it says.** `log(|x| + ε)` inverts the scale hierarchy of MiniLM embeddings. MiniLM is documented as anisotropic — a few high-magnitude dimensions dominate cosine similarity (Timkey & van Schijndel, 2021) — and the log transform compresses those and expands the low-magnitude residual dimensions. The fingerprint is a magnitude-blind per-dimension variance profile. This is mechanically true and interpretable.

- **It won a real elimination tournament.** On 188 clusters and 16,736 pairs, log-variance was the only transform among eleven that produced both a low false-positive rate (13.2%) and a non-zero set of cross-topic matches (23). The alternatives failed in informative ways — raw variance too permissive, squared kills all signal, rank matches everything. The comparison table is data, not assertion.

- **The discovery process was honest.** The concentration-of-measure hypothesis was tested via PCA and *disproved* — PCA made matching worse at every dimension count, redirecting the work to the correct (semantic-dominance) explanation. The XOR-parity failure and the squared-result collapse are internal consistency checks that came out as the theory predicted. Disproving your own first hypothesis on the record is worth more than any single positive result.

- **The retrieval decomposition is a clean idea.** Semantic (what), genealogical (where it came from), and structural (how) as three orthogonal retrieval axes is a defensible framing, and it rhymes with a real finding in cognitive science: people are bad at retrieving structural analogs across domains because surface similarity dominates recall (Gentner & Forbus, structure-mapping / MAC-FAC). A system that retrieves on structure is doing something unaided human memory does poorly. That's the framing to lead with.

---

## Claims to retire, downgrade, or hold as hypothesis

- **"Cognitive fingerprint" → "structural fingerprint."** This is the big one. "Epistemic posture" and "writing style" are confounded in text, and the residual embedding dimensions are exactly where register, syntax, and lexical statistics live. LogXform may be detecting *stylometry*, not cognition. Until the style/structure disentanglement experiment runs, the honest name is **structural fingerprint**, and the relevant prior art to distinguish from is authorship-attribution / stylometry, not just the isotropy literature. "Cognitive" is the hypothesis under test, not the finding.

- **"23/23 validated" → "23 matches, validation uncalibrated."** The 23 cross-topic matches were confirmed by an LLM, which is embedding-based and prone to finding methodological isomorphism between almost any two structured reasoning traces. A 23/23 confirmation rate means nothing without a null: the same judge, same prompt, on pairs that pass the semantic ceiling but fail the shape threshold. If the LLM confirms 70% of those too, the result is noise; if 5%, it's strong. The number to publish is the *separation from the null*, which doesn't exist yet.

- **"Substrate-agnostic" → hypothesis.** The result is MiniLM-specific until the same pairs match under a second embedding model (e.g. GTE-small). One cross-embedding replication run either confirms this or reveals MiniLM-specific artifacts. State it as a claim to be tested, not a property.

- **The Pribram / holonomic-brain connection → drop, or hold very loosely.** Holonomic brain theory makes specific claims about Fourier-domain encoding in dendritic networks that LogXform does not instantiate. "Distributed variance pattern as interference pattern" is an evocative intuition, not a literature position, and stating it as the latter invites exactly the wrong scrutiny. It's a framing device for the reader's intuition at most.

- **The ε = 1e-6 floor → flag as a known sensitivity.** `ln(1e-6) ≈ -13.8`, so dimensions hovering near zero can generate enormous log-variance from what may be numerical noise. Until an ε-sweep confirms the 23 matches survive ε ∈ [1e-3, 1e-8], the exact floor is an unvalidated parameter, not a principled constant.

- **Cluster-size confound → acknowledge until checked.** Variance estimated from 3 embeddings versus 30 has very different sampling noise. Until shape similarity is shown to be uncorrelated with node-count similarity across matched pairs, cluster size is a live confound.

---

## The one-line test for every claim in this repo

Before any sentence ships as a claim, ask:

> Is there a claim here not backed by a specific experimental result, a code inspection, or a cited passage? If yes, flag it — downgrade it to a hypothesis or cut it.

The Orchard's larger specification keeps a section that preserves its own earlier overclaiming verbatim, as a specimen of what it looks like when a writing process smooths a live open question into a confident answer. That habit applies here: the goal is not to sound finished. It's to be exactly as confident as the evidence, no more, so that when the null-baseline experiment runs and the result holds, there's nothing inflated standing next to it to discount it.
