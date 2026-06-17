#!/usr/bin/env python3
"""
LogXform — structural (epistemic-posture) fingerprinting over sentence embeddings.

This is a reference implementation of the method described in spec/logxform-spec.md.
It is a faithful port of the production Kotlin core (computeLogXformFingerprint),
reduced to the part that does not depend on The Orchard's knowledge graph: given a
set of text chunks, it produces a 384-float structural fingerprint and compares two
such fingerprints.

What it does, in one sentence: it reweights a sentence-embedding space so that the
low-magnitude "residual" dimensions — where structural/syntactic information lives —
dominate the variance, then fingerprints that variance pattern.

What it is NOT: a finished, validated cognitive-fingerprinting result. See FRAMING.md
for the distinction between what is demonstrated and what is still hypothesis, and
experiments/ for the controls that would settle it. Treat the output as a structural
similarity score, not a verdict about anyone's mind.

Usage:
    python logxform.py "first block of text" "second block of text"
    python logxform.py --file a.txt --file b.txt
    python logxform.py            # then paste two blocks at the prompts

Each text block is split into sentences; each sentence is embedded; the fingerprint is
the L2-normalized variance of the log-transformed embeddings across the block's
sentences. This mirrors the production path, where the "cluster" is a neighborhood of
graph nodes rather than the sentences of a single paste — the math is identical, the
unit of clustering differs. That difference is itself a thing to be honest about (see
the note in print_report).
"""

import argparse
import re
import sys

import numpy as np

# --- constants, matched to the production Kotlin core -----------------------
EMBEDDING_DIM = 384
LOG_EPSILON = 1e-6          # the ln(|x| + eps) floor; swept in experiments/eps_sweep.py
MODEL_NAME = "all-MiniLM-L6-v2"

# retrieval thresholds from the production spec (Section 3.2 / 4.1).
# they are reported here for reference; the CLI prints the raw scores and lets
# you judge, rather than returning a hard match/no-match, because the thresholds
# were tuned on graph-node clusters, not on sentence-split pastes.
SHAPE_THRESHOLD = 0.65
SEMANTIC_CEILING = 0.45


_model = None


def get_model():
    """Load MiniLM once. Imported lazily so --help is instant and the dependency
    is only required when you actually compute something."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit(
                "This needs sentence-transformers. Install it with:\n"
                "    pip install sentence-transformers\n"
            )
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def split_sentences(text: str) -> list[str]:
    """Cheap sentence split. Deliberately simple — the method is not sensitive to
    perfect segmentation, only to having several embeddings to take variance over.
    Falls back to line-splitting, then to the whole block, so it never returns empty
    on non-empty input."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 3]
    if len(parts) >= 2:
        return parts
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    if len(lines) >= 2:
        return lines
    return [text]


def embed(chunks: list[str]) -> np.ndarray:
    """Return an (n, 384) array of unit-norm MiniLM embeddings."""
    model = get_model()
    vecs = model.encode(chunks, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float64)


def logxform_fingerprint(embeddings: np.ndarray) -> np.ndarray:
    """The core method. Faithful port of computeLogXformFingerprint.

    1. log-transform every dimension of every embedding:  ln(|x| + eps)
    2. take the per-dimension variance across the chunk's embeddings
    3. L2-normalize the variance vector

    The log transform compresses the few large (semantic) dimensions and expands
    the many small (residual/structural) ones, so the variance ends up describing
    structural disagreement rather than topical content.
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"expected (n, {EMBEDDING_DIM}) embeddings, got {embeddings.shape}")

    transformed = np.log(np.abs(embeddings) + LOG_EPSILON)   # (n, 384)
    variance = transformed.var(axis=0)                        # (384,) population variance
    norm = np.linalg.norm(variance)
    if norm > 0 and np.isfinite(norm):
        variance = variance / norm
    return variance


def mean_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Unit-norm mean of the raw embeddings — used for the semantic-similarity
    ('what is this about') axis, the ceiling that enforces cross-domain matching."""
    m = embeddings.mean(axis=0)
    n = np.linalg.norm(m)
    if n > 0:
        m = m / n
    return m


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare(text_a: str, text_b: str) -> dict:
    """Compute both axes for two text blocks."""
    chunks_a, chunks_b = split_sentences(text_a), split_sentences(text_b)
    if not chunks_a or not chunks_b:
        raise ValueError("one of the inputs produced no usable text")

    emb_a, emb_b = embed(chunks_a), embed(chunks_b)
    fp_a, fp_b = logxform_fingerprint(emb_a), logxform_fingerprint(emb_b)
    mean_a, mean_b = mean_embedding(emb_a), mean_embedding(emb_b)

    shape = cosine(fp_a, fp_b)        # structural similarity (how)
    semantic = cosine(mean_a, mean_b) # topical similarity (what)
    return {
        "shape_similarity": shape,
        "semantic_similarity": semantic,
        "delta": shape - semantic,    # the discriminant: structurally alike, topically apart
        "n_chunks_a": len(chunks_a),
        "n_chunks_b": len(chunks_b),
    }


def semantic_analysis(text_a: str, text_b: str) -> dict:
    """A content-side analysis of the two blocks, separate from the structural
    fingerprint. This answers 'what are these about and how do they relate
    semantically' — the topical axis, examined rather than just scored.

    It reports, per block: sentence count, the per-sentence semantic spread
    (how internally varied each block is), and the most semantically-central
    sentence (the one closest to the block's own mean — its 'thesis').
    Across blocks it reports the single most semantically-aligned sentence pair
    and the single most divergent one, so you can see *where* the topical
    overlap and separation actually live, not just the aggregate cosine.
    """
    chunks_a, chunks_b = split_sentences(text_a), split_sentences(text_b)
    emb_a, emb_b = embed(chunks_a), embed(chunks_b)
    mean_a, mean_b = mean_embedding(emb_a), mean_embedding(emb_b)

    def central(chunks, embs, m):
        # cosine of each sentence to the block's own mean = how central it is
        sims = embs @ m
        i = int(np.argmax(sims))
        return chunks[i], float(sims.std())  # spread = std of within-block centrality

    central_a, spread_a = central(chunks_a, emb_a, mean_a)
    central_b, spread_b = central(chunks_b, emb_b, mean_b)

    # cross-block sentence-pair extremes
    cross = emb_a @ emb_b.T               # (na, nb) cosine matrix (unit vectors)
    ia, ib = np.unravel_index(int(np.argmax(cross)), cross.shape)
    da, db = np.unravel_index(int(np.argmin(cross)), cross.shape)

    return {
        "central_a": central_a, "spread_a": spread_a,
        "central_b": central_b, "spread_b": spread_b,
        "most_aligned": (chunks_a[ia], chunks_b[ib], float(cross[ia, ib])),
        "most_divergent": (chunks_a[da], chunks_b[db], float(cross[da, db])),
    }


def _truncate(s: str, n: int = 64) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def print_semantic_analysis(s: dict) -> None:
    print()
    print("  Semantic analysis (the 'what it's about' axis, examined)")
    print("  " + "-" * 48)
    print(f"  Block A thesis sentence (most central):")
    print(f"    \"{_truncate(s['central_a'])}\"")
    print(f"    internal spread {s['spread_a']:.3f}  (higher = more topically varied block)")
    print(f"  Block B thesis sentence (most central):")
    print(f"    \"{_truncate(s['central_b'])}\"")
    print(f"    internal spread {s['spread_b']:.3f}")
    print()
    ma, mb, msim = s["most_aligned"]
    da, db, dsim = s["most_divergent"]
    print(f"  Closest sentence pair (semantic {msim:+.3f}):")
    print(f"    A: \"{_truncate(ma)}\"")
    print(f"    B: \"{_truncate(mb)}\"")
    print(f"  Most divergent pair (semantic {dsim:+.3f}):")
    print(f"    A: \"{_truncate(da)}\"")
    print(f"    B: \"{_truncate(db)}\"")
    print("  " + "-" * 48)
    print("  Note: this is content (topic) analysis. It is deliberately separate")
    print("  from the structural fingerprint — LogXform's claim is that structure")
    print("  and content are different axes, so the tool shows both, not a blend.")
    print()


def print_report(r: dict) -> None:
    print()
    print("  LogXform comparison")
    print("  " + "-" * 48)
    print(f"  shape similarity     {r['shape_similarity']:+.3f}   (structural: how it's reasoned)")
    print(f"  semantic similarity  {r['semantic_similarity']:+.3f}   (topical: what it's about)")
    print(f"  delta (shape - sem)  {r['delta']:+.3f}   (high = same shape, different topic)")
    print("  " + "-" * 48)

    shape, semantic = r["shape_similarity"], r["semantic_similarity"]
    if shape > SHAPE_THRESHOLD and semantic < SEMANTIC_CEILING:
        print(f"  Reading: structurally similar AND topically distinct.")
        print(f"  This is the signal LogXform targets (cross-domain structural resonance).")
    elif shape > SHAPE_THRESHOLD and semantic >= SEMANTIC_CEILING:
        print(f"  Reading: structurally similar, but also topically related — the production")
        print(f"  pipeline's semantic ceiling ({SEMANTIC_CEILING}) would reject this as 'just same topic.'")
    else:
        print(f"  Reading: structurally dissimilar by the {SHAPE_THRESHOLD} threshold.")

    print()
    print("  Caveats, stated plainly:")
    print("  - Thresholds were tuned on graph-node clusters, not sentence-split pastes;")
    print("    treat the raw scores as the answer, not the verdict line above.")
    print("  - 'Shape' here is structural, which may be cognitive OR merely stylometric.")
    print("    Distinguishing those is an open experiment (see FRAMING.md).")
    print(f"  - n chunks: {r['n_chunks_a']} vs {r['n_chunks_b']}. Very small or very")
    print("    uneven counts make the variance estimate noisy.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute LogXform structural similarity between two text blocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", nargs="*", help="two text blocks (quote each one)")
    p.add_argument("--file", action="append", default=[], metavar="PATH",
                   help="read a block from a file (use twice)")
    p.add_argument("--analyze", action="store_true",
                   help="also print a semantic content analysis of the two blocks")
    args = p.parse_args()

    blocks: list[str] = []
    for path in args.file:
        with open(path, encoding="utf-8") as f:
            blocks.append(f.read())
    blocks.extend(args.text)

    if len(blocks) == 0:
        print("Paste the FIRST block, then press Ctrl-D (Ctrl-Z on Windows):")
        a = sys.stdin.read()
        print("Paste the SECOND block, then press Ctrl-D:")
        b = sys.stdin.read()
        blocks = [a, b]

    if len(blocks) != 2:
        sys.exit(f"need exactly two text blocks, got {len(blocks)}")

    print_report(compare(blocks[0], blocks[1]))
    if args.analyze:
        print_semantic_analysis(semantic_analysis(blocks[0], blocks[1]))


if __name__ == "__main__":
    main()
