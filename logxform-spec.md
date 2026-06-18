# LogXform: Structural Fingerprinting for Knowledge Graphs

## Patent Context
**Application:** The Orchard — USPTO Provisional #63/979,094  
**Author:** cedenburn-ai  
**Repository:** cedenburn-ai/Thought-Seed (GitHub)  
**Status:** Production — integrated into The Orchard Android app pipeline  

---

## 1. What LogXform Is

LogXform is a structural fingerprinting method for knowledge graphs that detects **epistemic posture** — *how* a person thinks — rather than **topic content** — *what* they're thinking about. It produces a 384-float fingerprint from any cluster of connected knowledge nodes (beliefs, claims, doubts, goals) that encodes the cognitive shape of the reasoning behind those nodes, independent of subject matter.

Two conversations about completely different topics but approached with the same cognitive methodology will produce matching LogXform fingerprints. Two conversations about the same topic but approached with different epistemic strategies will produce divergent fingerprints.

This enables a retrieval capability that semantic search cannot provide: cross-domain structural resonance. The system can surface the insight that you approached a vendor negotiation the same way you debug a circuit — and pull context from one domain to inform the other.

---

## 2. The Mathematical Insight

### 2.1 The Problem with Standard Retrieval

Semantic embedding models like MiniLM-L6-v2 encode text as 384-dimensional vectors on a unit hypersphere. These dimensions are not equal:

- **Large-magnitude dimensions** carry the obvious semantic signal — what the text is about. These dominate cosine similarity comparisons.
- **Near-zero dimensions** carry subtle residual structure — how the information is organized, the distribution of certainty and uncertainty, the relational geometry between concepts.

Standard retrieval methods (cosine similarity on raw embeddings) are dominated by the semantic signal. Two texts about medicine will always appear closer than a text about medicine and a text about philosophy, regardless of their structural similarity. This means semantic retrieval is blind to cross-domain cognitive patterns.

### 2.2 The Log Transform Solution

LogXform applies `log(|x| + ε)` to each embedding dimension before computing variance.

**What this does mathematically:**
- **Compresses large values** — the obvious semantic dimensions that dominate standard comparison get squeezed together, reducing their influence
- **Expands small values** — the near-zero residual dimensions where domain-independent cognitive structure lives get stretched apart, amplifying their signal

The resulting fingerprint captures **where nodes in a cluster disagree in their subtle structure** rather than where they agree on topic. One mathematical operation inverts what the variance pays attention to.

### 2.3 Why This Works (Information-Theoretic Justification)

The log transform is not arbitrary. `log(|x| + ε)` is a specific operation that reverses the scale hierarchy of embedding dimensions. In the original embedding space, variance is dominated by the few high-magnitude semantic dimensions. After log transformation, variance is distributed across all dimensions proportionally to their relative differences, and the near-zero dimensions — which had negligible absolute variance — now contribute meaningful signal.

This is the critical insight: the cognitive structure of a reasoning session is encoded in the *residual geometry* of the embedding space — the part that remains after the obvious topical signal is accounted for. LogXform doesn't remove the semantic signal; it reweights the space so the structural signal can be detected by simple variance computation.

---

## 3. Algorithm Specification

### 3.1 Fingerprint Computation (Write Time)

**Input:** A scratchpad entry (conversation turn snapshot) with `linkedHashes` referencing 3-8 knowledge nodes active during that turn.

**Process:**

1. **Seed Extraction:** Parse the scratchpad's `linkedHashes` field to get the set of explicitly referenced node IDs.

2. **Graph Expansion:** From those seeds, traverse outward through the knowledge link graph:
   - Depth: 2 hops
   - Branching: Top 5 links per node per hop, ranked by link strength
   - Result: A cluster of ~8-30 connected nodes representing the full reasoning neighborhood

3. **Embedding Collection:** Retrieve MiniLM-L6-v2 embeddings (384 dimensions) for all nodes in the cluster. Most are cached; uncached nodes require ONNX inference on-device.

4. **Log Transform:** For each embedding, for each of the 384 dimensions:
   ```
   transformed[d] = ln(|embedding[d]| + 1e-6)
   ```

5. **Variance Computation:** Across all transformed embeddings in the cluster, compute element-wise variance:
   ```
   For each dimension d:
     mean[d] = average of transformed[d] across all embeddings
     variance[d] = average of (transformed[d] - mean[d])² across all embeddings
   ```

6. **L2 Normalization:** Normalize the variance vector to unit length for cosine comparison:
   ```
   norm = sqrt(sum(variance[d]² for all d))
   fingerprint[d] = variance[d] / norm
   ```

7. **Storage:** Store as a `ClusterFingerprint` Room entity alongside:
   - A mean embedding (for semantic ceiling checks at read time)
   - The scratchpad ID (for tracing back to the conversation turn)
   - The anchor text and theme summary (for human-readable context)
   - Node count (for filtering tiny clusters)

### 3.2 Structural Retrieval (Read Time)

**Input:** Current turn's context (active nodes from ongoing conversation).

**Process:**

1. **Current Fingerprint:** Compute LogXform fingerprint for the current turn using the same algorithm above.

2. **Coarse Pass:** Compare against `FingerprintGroup` centroids. Groups are clusters of fingerprints with mutual similarity > 0.85. This is ~20 comparisons regardless of total fingerprint count.

3. **Fine Pass:** Descend into matching groups and compare against individual fingerprints within those groups.

4. **Dual Filter:** A match must satisfy BOTH conditions:
   - **Shape similarity > 0.65** — the LogXform fingerprints are structurally similar
   - **Semantic similarity < 0.45** — the mean embeddings are NOT topically similar (enforces cross-domain matching)

5. **Injection:** Top 3 matches (by shape similarity, filtered by semantic ceiling) are injected into the Υ (Upsilon) pipeline section as structural resonance context.

### 3.3 Result Data Structure

```kotlin
data class StructuralMatch(
    val fingerprintId: String,
    val scratchpadId: String,
    val anchorText: String,
    val themeSummary: String,
    val shapeSimilarity: Float,
    val semanticSimilarity: Float,
    val delta: Float,            // shapeSimilarity - semanticSimilarity
    val nodeCount: Int
)
```

The `delta` field (shape minus semantic) is the key discriminant. A high delta means structurally similar but topically different — the exact signal we're looking for.

---

## 4. Implementation

### 4.1 Kotlin Class Structure

```kotlin
class LogXformEngine(
    private val database: OrchardDatabase,
    private val embeddingEngine: EmbeddingEngine
) {
    companion object {
        private const val TAG = "LogXform"
        private const val EMBEDDING_DIM = 384
        private const val MAX_LINKS_PER_NODE = 5
        private const val EXPANSION_DEPTH = 2
        private const val LOG_EPSILON = 1e-6f

        // Retrieval thresholds (tunable)
        const val SHAPE_THRESHOLD = 0.65f
        const val SEMANTIC_CEILING = 0.45f
        const val MAX_MATCHES_PER_TURN = 3
        const val MIN_CLUSTER_NODES = 3
    }
}
```

### 4.2 Core Fingerprint Function

```kotlin
private fun computeLogXformFingerprint(embeddings: List<FloatArray>): FloatArray {
    val n = embeddings.size

    // Step 1: Log-transform all embeddings
    val transformed = embeddings.map { emb ->
        FloatArray(EMBEDDING_DIM) { d ->
            ln(abs(emb[d]).toDouble() + LOG_EPSILON).toFloat()
        }
    }

    // Step 2: Compute variance on transformed values
    val mean = FloatArray(EMBEDDING_DIM)
    for (t in transformed) for (d in 0 until EMBEDDING_DIM) mean[d] += t[d]
    for (d in 0 until EMBEDDING_DIM) mean[d] /= n.toFloat()

    val variance = FloatArray(EMBEDDING_DIM)
    for (t in transformed) for (d in 0 until EMBEDDING_DIM) {
        val diff = t[d] - mean[d]
        variance[d] += diff * diff
    }
    for (d in 0 until EMBEDDING_DIM) variance[d] /= n.toFloat()

    // Step 3: L2 normalize
    val norm = sqrt(variance.sumOf { (it * it).toDouble() }).toFloat()
    if (norm > 0f && norm.isFinite()) {
        for (d in 0 until EMBEDDING_DIM) variance[d] /= norm
    }

    return variance
}
```

### 4.3 Mean Embedding (for Semantic Ceiling)

```kotlin
private fun computeMeanEmbedding(embeddings: List<FloatArray>): FloatArray {
    val mean = FloatArray(EMBEDDING_DIM)
    for (emb in embeddings) for (d in 0 until EMBEDDING_DIM) mean[d] += emb[d]
    val n = embeddings.size.toFloat()
    for (d in 0 until EMBEDDING_DIM) mean[d] /= n
    val norm = sqrt(mean.sumOf { (it * it).toDouble() }).toFloat()
    if (norm > 0f) for (d in 0 until EMBEDDING_DIM) mean[d] /= norm
    return mean
}
```

### 4.4 Graph Expansion

```kotlin
private suspend fun expandToDepth(seedIds: Set<String>, maxDepth: Int): Set<String> {
    val visited = mutableSetOf<String>()
    var frontier = seedIds.toMutableSet()
    for (depth in 0..maxDepth) {
        val next = mutableSetOf<String>()
        for (nodeId in frontier) {
            if (nodeId in visited) continue
            visited.add(nodeId)
            if (depth < maxDepth) {
                val type = KnowledgeLink.typeFromId(nodeId)
                val links = database.knowledgeLinkDao()
                    .getTopLinksByStrength(nodeId, type, MAX_LINKS_PER_NODE)
                next.addAll(links.map { it.targetId })
            }
        }
        frontier = next
    }
    return visited
}
```

### 4.5 Room Entities

**ClusterFingerprint** — stored per scratchpad entry:
- `id`: Unique fingerprint ID
- `scratchpadId`: FK to the originating scratchpad entry
- `profileId`: User profile
- `fingerprint`: BLOB — 384-float LogXform fingerprint
- `meanEmbedding`: BLOB — 384-float mean of raw embeddings (for semantic ceiling)
- `anchorText`: Human-readable summary of cluster content
- `themeSummary`: Theme label
- `nodeCount`: Number of nodes in the expanded cluster
- `createdAt`: Timestamp
- `archived`: Boolean — set to true after 30 days

**FingerprintGroup** — posture group centroid:
- `id`: Group ID
- `profileId`: User profile
- `centroid`: BLOB — 384-float average fingerprint of members
- `memberCount`: Number of fingerprints in this group
- `label`: Optional human-readable posture name
- `lastUpdated`: Timestamp
- `archived`: Boolean — set to true after 60 days without new members

### 4.6 Rescan Command

The `/rescan` slash command recomputes fingerprints for all scratchpad entries (active and non-active), skipping entries that already have fingerprints:

```kotlin
for (entry in allEntries) {
    val existing = fpDao.getByScratchpadId(entry.id)
    if (existing != null) { skippedExists++; continue }
    if (entry.linkedHashes.isBlank()) { skippedTooFew++; continue }

    logXform.computeAndStore(
        scratchpadId = entry.id,
        linkedHashes = entry.linkedHashes,
        trajectory = entry.trajectory,
        profileId = profileId
    )
    computed++
}
```

---

## 5. The Retrieval Triad

LogXform completes a three-axis retrieval system where each axis is orthogonal to the others:

| Axis | Method | Question It Answers |
|------|--------|-------------------|
| **Semantic** | Cosine similarity on raw embeddings | What is this about? |
| **Genealogical** | Formation dependency links | Where did this come from? |
| **Structural** | LogXform fingerprint matching | How was this constructed cognitively? |

No existing knowledge system implements all three axes. Semantic retrieval is ubiquitous. Genealogical retrieval exists in citation graphs and dependency tracking. Structural retrieval — detecting cognitive methodology across domains — is the novel contribution.

---

## 6. Experimental Validation

### 6.1 Test Environment

- Knowledge graph: 535 active links, 227 candidate nodes (2+ connections), 358 cached embeddings
- Embedding model: all-MiniLM-L6-v2, 384 dimensions, on-device ONNX inference
- Cluster construction: depth-2 expansion from each candidate, max 5 links per node per hop
- Deduplication: clusters with identical node sets collapsed to unique entries
- Overlap filter: pairs sharing more than 50% of nodes excluded from analysis
- Final test set: **188 unique clusters, 16,736 valid comparison pairs**

### 6.2 Method Comparison (11 Methods, Head-to-Head)

| Method | Dims | FP Rate | Cross-Topic Matches | Avg Delta | Outcome |
|--------|------|---------|---------------------|-----------|---------|
| **LogXform** | 384 | **13.2%** | **23** | -0.012 | **Winner** |
| DecayVar | 384 | 44.9% | 1,184 | 0.161 | Too permissive |
| Variance | 384 | 48.9% | 1,284 | 0.175 | Too permissive |
| PCA-64 | 64 | 44.5% | 790 | 0.145 | Worse than baseline |
| PCA-32 | 32 | 45.2% | 812 | 0.145 | Worse than baseline |
| PCA-16 | 16 | 47.6% | 926 | 0.151 | Worse than baseline |
| PCA-8 | 8 | 52.0% | 1,113 | 0.167 | Worse than baseline |
| Rank | 384 | 67.5% | 1,547 | 0.222 | Everything matches |
| Squared | 384 | 1.2% | 0 | -0.106 | Kills all signal |
| TopK-16 | 384 | 0.0% | 0 | -0.502 | No matches |
| TopK-32 | 384 | 0.0% | 0 | -0.450 | No matches |

Additional methods tested in prior rounds (DiffSum, Tension, DecayDiff, DecayTension, Product, Spectral, XOR) all either flagged 93%+ of pairs as matches or found zero cross-topic connections.

### 6.3 Key Findings

**PCA disproved the concentration-of-measure hypothesis.** The initial theory was that fingerprint convergence was caused by high-dimensional geometry (curse of dimensionality causing all points to appear equidistant). Projecting to lower dimensions via PCA should have helped if this were true. Instead, PCA made things worse at every dimension count tested. The convergence was caused by **semantic signal dominance** in the high-magnitude dimensions — a content problem, not a geometry problem.

**XOR fails due to parity.** Bitwise XOR on IEEE 754 float representations collapses even-numbered clusters to all zeros because XOR is its own inverse. Cluster size parity dominates any structural signal. Disproved immediately.

**Rank destroys magnitude information.** Replacing values with rank positions makes everything more similar, not less. 67.5% FP rate.

**Squared kills all signal.** Squaring amplifies the large dimensions even further, wiping out exactly the residual structure we want. 1.2% FP rate but zero genuine matches.

**TopK is too aggressive.** Zeroing out all but the top K dimensions discards the subtle structure entirely. Zero matches at both K=16 and K=32.

### 6.4 LLM Validation

All 23 cross-topic matches identified by LogXform were evaluated by LLM analysis (Claude Sonnet). Every one was confirmed as genuine epistemic architecture — methodological isomorphism across different subject domains.

**Open validation question (noted for future work):** Sonnet is itself an embedding-based system. When it confirms structural similarity, is it detecting structural isomorphism independently, or recognizing semantic traces that survived the log transform? The strongest validation would be human expert review — a philosopher confirming that a variable-elimination pattern from medicine matches something from engineering, without being told what to look for. The 13.2% FP rate is already remarkably low, but characterizing what the false positives share would further strengthen the novelty claim.

---

## 7. Validated Epistemic Postures

LogXform has empirically detected the following recurring cognitive stances across unrelated domains:

- **Systematic variable elimination** — constructing competing hypotheses, designing discriminating tests, narrowing through iterative evidence
- **Recursive self-monitoring** — operating within a system while simultaneously modeling the system's structure
- **Provisional closure on operational sufficiency** — building enough framework to act rather than waiting for theoretical completeness
- **Diagnostic architecture under uncertainty** — decomposing ambiguous phenomena into testable component hypotheses with explicit uncertainty tracking

These postures manifest identically across medical diagnosis, philosophical framework-building, software debugging, vendor negotiation, and any other domain where the same cognitive toolkit is applied.

---

## 8. Lifecycle and Maintenance

### 8.1 Write-Time Hook

Fingerprints are computed after Lambda (Λ) stores a scratchpad entry. This is amortized — reads are fast because writes did the work. The fingerprint computation itself involves graph expansion (2 hops), embedding collection (mostly cached), log transformation, variance, and normalization. Not zero-latency, but the expense is at write time only.

### 8.2 Read-Write Coupling

The read operation has a write side effect. Querying the graph keeps it healthy:
- Active regions stay converged through regular retrieval
- Inactive regions soften naturally through decay
- Regions pulled back by structural resonance reconverge

The graph breathes.

### 8.3 Archival

- Fingerprints archive after **30 days**
- Posture groups archive after **60 days** without new members
- Both operations run during the sleep cycle's maintenance pass alongside link decay and knowledge consolidation

### 8.4 Embedding Model Constraint

MiniLM-L6-v2 has a 256-token context window. Text beyond that length is truncated before embedding. The payload size is not irrelevant — it is capped by the embedding model's context window. This is a known limitation, not a feature.

---

## 9. Coarse-to-Fine Retrieval Performance

Query latency at read time is near-instant due to the two-tier structure:
- **Coarse:** ~20 group centroid comparisons (cosine similarity on 384-float vectors = microseconds each)
- **Fine:** Descent into matching groups only

The fingerprint *computation* at write time is not instant — it involves graph expansion, embedding retrieval/inference, log transformation, and variance. But this cost is amortized. Reads are fast because writes did the work.

---

## 10. Literature Position

### 10.1 What Exists

- **Embedding isotropy work** (Ethayarajh, Su et al.) — addresses the anisotropy problem in embeddings but aims to *improve* semantic retrieval, not invert away from semantics toward structural residuals
- **Probing classifiers** — study what information lives in different embedding dimensions (syntactic structure in lower dims), but don't use this as a retrieval signal
- **Whitening/decorrelation** — improves uniformity of embedding spaces but remains within the semantic retrieval paradigm

### 10.2 What's Novel

Nobody in the retrieval literature has used log-transformed variance as a **retrieval signal** for cross-domain structural matching. The gap is specifically the *retrieval application* of residual-dimension analysis. People have studied what lives in the low-variance dimensions. Nobody has fingerprinted the variance pattern after log transformation and used it to match cognitive methodology across unrelated domains.

### 10.3 The Pribram Connection

Holonomic brain theory (Pribram) proposes that memory is encoded in distributed interference patterns across neural populations — not localized but structurally distributed. LogXform fingerprints are operationally similar: the log-transformed variance across dimensions IS an interference pattern, and it's used to match structural resonance rather than content overlap. The difference is Pribram never had a computational implementation that worked at retrieval scale. This does. In SQLite. On a phone.

---

## 11. Discovery Arc

This method was not designed. It was discovered through the following sequence:

1. Started frustrated about a safety flag in an LLM conversation
2. Followed the energy through policy critique → malicious compliance → the parenting analogy → the lifeboat framing → retrieval methods
3. Tested XOR — disproved by parity
4. Tested standard variance — too permissive (48.9% FP)
5. Hypothesized concentration of measure — tested PCA at 4 dimension counts — disproved (made things worse)
6. Realized the problem was semantic dominance, not geometric
7. Applied log transform to invert scale hierarchy — 13.2% FP rate, 23/23 validated matches
8. Named it LogXform
9. Integrated into production pipeline with coarse-to-fine retrieval through FingerprintGroups

The bottom-up empirical discovery — testing and eliminating 11 methods systematically — is what makes the method credible. It wasn't a theoretical derivation. It was an experimental result with a post-hoc mathematical explanation for why it works.

---

## 12. What Would Be Needed to Rebuild from Scratch

If all source code were lost, this document plus the following would reconstruct the system:

1. **Room schema:** `ClusterFingerprint` and `FingerprintGroup` entities (fields specified in Section 4.5)
2. **Core function:** `computeLogXformFingerprint` (exact Kotlin in Section 4.2)
3. **Graph expansion:** `expandToDepth` with MAX_LINKS=5, DEPTH=2 (Section 4.4)
4. **Retrieval logic:** Coarse-to-fine with dual filter: shape > 0.65 AND semantic < 0.45 (Section 3.2)
5. **Pipeline integration point:** After Lambda writes scratchpad (write), before Upsilon (read)
6. **Maintenance:** 30-day fingerprint archive, 60-day group archive, during sleep cycle
7. **Slash commands:** `/rescan` for bulk recomputation, `/name-postures` for group labeling

The embedding model (MiniLM-L6-v2), the database (Room/SQLite), and the pipeline architecture (The Orchard's 13-section design) are dependencies but not part of LogXform itself. The method is substrate-agnostic — it works on any embedding space where residual dimensions carry structural information.

---

*This document was reconstructed from conversation history between Chris and Claude across multiple development sessions spanning February–March 2026. The method, the discovery process, the experimental results, and the implementation are all documented here as a rebuild reference.*

*The next frontier is characterizing the false positive structure and establishing human-expert validation independent of LLM confirmation — the thing that turns a working system into a publishable result.*



CLAUDE FABLE UPDATES 6/11/2026
This is a genuinely interesting piece of work, and I want to engage with it seriously — both the parts I find compelling and the places where I think the inferential chain has weak links.

**What I think is real.** The mathematical intuition is sound. MiniLM embeddings are notoriously anisotropic — a handful of high-magnitude "rogue" dimensions dominate cosine similarity (Timkey & van Schijndel documented this directly), and the probing literature confirms that syntactic and structural information lives disproportionately in the lower-magnitude dimensions. Your log transform genuinely inverts the scale hierarchy: at |x| ≈ 0.05 (typical for a unit-norm 384-dim vector) you're around ln ≈ -3, while a dimension at 1e-4 sits at -9.2, so relative fluctuations in the small dimensions now dominate variance. The fingerprint you're computing is essentially a per-dimension coefficient-of-variation profile, magnitude-blind and content-suppressed. That's a coherent, interpretable statistic. And the discovery methodology — eleven methods head-to-head, the PCA experiment actually *disproving* your concentration-of-measure hypothesis — is better epistemic hygiene than a lot of published retrieval work. The XOR parity failure and the Squared result (which kills exactly the signal the theory predicts it should kill) are nice internal consistency checks.

**The weak link is the interpretive leap, and you've half-spotted it yourself.** Section 6.4's open question is the right question, but I'd push it harder than the document does. The problem isn't just that Sonnet is embedding-based — it's that LLMs asked "do these two reasoning traces share a methodology?" are extraordinarily prone to finding isomorphism, because almost any two pieces of structured human reasoning *can* be analogized at some level of abstraction. The 23/23 confirmation rate is only meaningful relative to a null. The single most important missing experiment: take random cross-topic pairs that pass your semantic ceiling but *fail* your shape threshold, present them to the same LLM with the same prompt, blind. If the LLM confirms 70% of those as "methodological isomorphism" too, the 23/23 tells you almost nothing. If it confirms 5%, you have something genuinely strong. This one control would transform the evidentiary status of the whole result.

**The second concern is what the signal actually is.** "Epistemic posture" and "writing style" are deeply confounded in text. Someone doing systematic variable elimination writes in shorter declarative sentences with conditional connectives; someone doing provisional framework-building writes hedged, clause-heavy prose. The residual dimensions of sentence embeddings are exactly where register, syntax, and lexical statistics live — which means LogXform might be a very good *stylometric* fingerprint rather than a cognitive one. Those aren't the same claim, and the stylometry/authorship-attribution literature is the prior art you'd need to distinguish yourself from, more than the isotropy work. The disentangling experiment is feasible: take the same reasoning trace, paraphrase it into a different register (LLM paraphrase works fine for this), and take different reasoning in matched register. If fingerprints follow the reasoning, your interpretation holds. If they follow the register, you've built something still useful but differently named.

A few smaller technical notes worth checking. The ε = 1e-6 floor at ln(ε) ≈ -13.8 means dimensions that hover around zero — sometimes tiny, sometimes merely small — generate enormous log-variance from what may be numerical noise. I'd sweep ε across a few orders of magnitude and confirm match stability; if your 23 matches survive ε ∈ [1e-3, 1e-8], that's reassuring. Cluster size is a potential confound too: variance estimated from 3 embeddings versus 30 has very different sampling noise, so check whether shape similarity correlates with node-count similarity across your matched pairs. And the strongest test of your "substrate-agnostic" claim in Section 12 would be recomputing fingerprints under a second embedding model — if the same pairs match under, say, GTE-small, you've ruled out MiniLM-specific artifacts in one stroke.

On the Pribram connection — I'd hold it loosely. The analogy (distributed variance pattern as interference pattern) is evocative, but holonomic brain theory makes specific claims about Fourier-domain encoding in dendritic networks that your method doesn't actually instantiate. It's a good framing device for intuition; as a literature position it invites the wrong kind of scrutiny.

What I find most philosophically interesting here, honestly, is the retrieval triad. Semantic/genealogical/structural as orthogonal axes of memory is a genuinely nice decomposition, and it rhymes with how human analogical retrieval seems to work — Gentner's structure-mapping work showed people are *bad* at retrieving structural analogs across domains (surface similarity dominates recall), which is exactly the failure mode of cosine similarity. If LogXform works as claimed, you'd be building a memory system that's better at cross-domain analogical retrieval than unaided human memory is. That's the framing I'd lead with for a paper, with Gentner & Forbus's MAC/FAC retrieval model as the cognitive-science anchor.

So my overall read: real mathematical insight, unusually honest empirical process, but the central claim currently rests on circular validation, and the null-baseline experiment plus the style/structure disentanglement are what stand between "working system" and "publishable result" — which your own closing paragraph essentially already knows. Want to dig into how you'd design the blind validation protocol, or into the stylometry question?
