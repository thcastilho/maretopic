# SPDX-License-Identifier: GPL-2.0
# MARETopic — embeddings.py
# SBERT encoding and ranked-list construction.
# UMAP belongs to the MARETopic class (it is model state, not a utility).

from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree

# Module-level model cache: avoids re-loading (and re-allocating on MPS/CUDA)
# the same SentenceTransformer weights on every encode_sbert() call.
_SBERT_CACHE: dict[str, "SentenceTransformer"] = {}  # type: ignore[type-arg]


def _get_sbert_model(model_name: str) -> "SentenceTransformer":  # type: ignore[type-arg]
    """Return a cached SentenceTransformer instance, loading on first use."""
    if model_name not in _SBERT_CACHE:
        from sentence_transformers import SentenceTransformer
        _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    return _SBERT_CACHE[model_name]


def encode_sbert(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: str | Path | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts using a Sentence-Transformers model.

    The model is cached at module level so repeated calls (e.g. across multiple
    stochastic runs) reuse the same weights without re-allocating GPU memory.

    Parameters
    ----------
    texts : list of str
        Documents to encode.
    model_name : str
        Sentence-Transformers model identifier.
    cache_path : path-like, optional
        If given, load from cache when available; save after encoding.
    batch_size : int
        Encoding batch size. Defaults to 64 to keep MPS peak allocation low;
        increase if you have headroom (e.g. 128 on machines with >24 GB RAM).
    show_progress : bool
        Show tqdm progress bar during encoding.

    Returns
    -------
    embeddings : np.ndarray of shape (N, D), dtype float32
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path)

    model = _get_sbert_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Release MPS intermediate buffers immediately after encoding so they
    # don't accumulate across multiple runs of the same experiment.
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings


def build_ranked_lists(
    embeddings: np.ndarray,
    top_K: int,
) -> list[list[int]]:
    """
    Build ranked lists using a BallTree k-NN index.

    For each document i, returns the top_K most similar documents
    (excluding i itself), sorted by ascending distance (= descending similarity).

    BallTree uses Euclidean distance on L2-normalised embeddings, which
    produces the same ranking as cosine similarity on the original vectors
    (on unit vectors: ||u-v||² = 2 - 2·cos(u,v)).

    Parameters
    ----------
    embeddings : np.ndarray of shape (N, D)
    top_K : int
        Neighbourhood depth.

    Returns
    -------
    ranked_lists : list of list of int, length N, each inner list of length top_K
    """
    N = len(embeddings)
    k = min(top_K + 1, N)  # +1 because BallTree returns self at rank 0

    # L2-normalise so Euclidean distance ≡ cosine distance ranking
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)

    normed = (embeddings / norms).astype(np.float64)
    tree = BallTree(normed)
    _, indices = tree.query(normed, k=k)

    # Drop self (column 0) and truncate to top_K
    ranked_lists = [row[1:top_K + 1].tolist() for row in indices]
    return ranked_lists
