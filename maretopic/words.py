# SPDX-License-Identifier: GPL-2.0
# MARETopic — words.py
# Per-topic keyword extraction:
#   c-TF-IDF (BERTopic formula) + inter-topic MMR for diversity.

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def _compute_ctfidf_and_select(
    cluster_assignments: np.ndarray,
    texts: list[str],
    num_top_words: int = 15,
    mmr_diversity: float = 0.3,
) -> list[list[tuple[str, float]]]:
    """
    Core c-TF-IDF + MMR computation, returning words with their scores.

    Pipeline
    --------
    1. Concatenate documents per topic cluster.
    2. Compute c-TF-IDF (BERTopic formula):
           W_{t,c} = tf_{t,c} * log(1 + A / tf_c)
       where tf_{t,c} = term frequency of word c in topic t,
             tf_c     = total frequency of word c across all topics,
             A        = average number of words per topic.
    3. Apply MMR inter-topic deduplication: topics processed largest-first;
       previously selected words are penalized for subsequent topics.

    Parameters
    ----------
    cluster_assignments : np.ndarray of shape (N,), dtype int
        Hard topic assignment per training document (argmax of theta).
    texts : list of str
        Training documents (preprocessed — same corpus used for c-TF-IDF).
    num_top_words : int
        Number of keywords to extract per topic.
    mmr_diversity : float in [0, 1]
        MMR diversity weight.
        0.0 → pure c-TF-IDF (no inter-topic deduplication).
        0.3 → recommended sweet spot (large TD gain, minimal C_v cost).
        1.0 → maximum deduplication (words never repeated across topics).

    Returns
    -------
    topic_word_scores : list of list of (word, score), length K
        Each inner list has ``num_top_words`` tuples of (word, normalized
        c-TF-IDF score ∈ [0, 1]).

    Notes
    -----
    Purity and NMI are independent of MMR (they depend only on cluster_assignments).
    C_v and TD are affected by word selection. The diversity=0.3 sweet spot was
    validated in Phase 3 grid: TD 0.608 → 0.863 with C_v cost of only −0.002.
    """
    K = int(cluster_assignments.max()) + 1

    # --- Build per-topic text blobs and sizes ---
    cluster_texts: list[str] = []
    cluster_sizes: list[int] = []
    for k in range(K):
        mask = cluster_assignments == k
        cluster_sizes.append(int(mask.sum()))
        cluster_texts.append(
            " ".join(texts[i] for i in range(len(texts)) if mask[i])
        )

    # --- Count matrix (K × V) ---
    vectorizer = CountVectorizer()
    tf_matrix = vectorizer.fit_transform(cluster_texts).toarray().astype(np.float64)
    vocab = vectorizer.get_feature_names_out()
    V = len(vocab)

    # --- c-TF-IDF (BERTopic formula) ---
    tf_c = tf_matrix.sum(axis=0)           # total word counts across topics
    A = tf_matrix.sum(axis=1).mean()       # average words per topic
    ctfidf = tf_matrix * np.log(1.0 + A / np.maximum(tf_c, 1.0))

    # Normalize per topic (scores in [0, 1]) for MMR comparability
    for k in range(K):
        mx = ctfidf[k].max()
        if mx > 0:
            ctfidf[k] /= mx

    # --- MMR inter-topic deduplication (largest topics first) ---
    topic_order = np.argsort(-np.array(cluster_sizes))
    word_penalty = np.zeros(V, dtype=np.float64)
    topic_word_scores: list[list[tuple[str, float]] | None] = [None] * K

    for k in topic_order:
        scores = (1.0 - mmr_diversity) * ctfidf[k] - mmr_diversity * word_penalty
        top_indices = np.argsort(-scores)[:num_top_words]
        # Capture words + their *original* c-TF-IDF scores (not MMR-adjusted)
        topic_word_scores[k] = [
            (str(vocab[idx]), float(ctfidf[k, idx])) for idx in top_indices
        ]
        # Update penalty: track max c-TF-IDF seen for each selected word
        for idx in top_indices:
            word_penalty[idx] = max(word_penalty[idx], ctfidf[k, idx])

    return topic_word_scores  # type: ignore[return-value]


def extract_topic_words(
    cluster_assignments: np.ndarray,
    texts: list[str],
    num_top_words: int = 15,
    mmr_diversity: float = 0.3,
) -> list[str]:
    """
    Extract topic keywords via c-TF-IDF with inter-topic MMR deduplication.

    Returns
    -------
    top_words : list of str, length K
        Each element is a space-separated string of keywords for that topic.
    """
    scored = _compute_ctfidf_and_select(
        cluster_assignments, texts, num_top_words, mmr_diversity
    )
    return [" ".join(word for word, _ in ws) for ws in scored]


def extract_topic_words_with_scores(
    cluster_assignments: np.ndarray,
    texts: list[str],
    num_top_words: int = 15,
    mmr_diversity: float = 0.3,
) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """
    Extract topic keywords with their normalized c-TF-IDF scores.

    Returns
    -------
    top_words : list of str, length K
        Each element is a space-separated string of keywords for that topic.
        Identical to the output of ``extract_topic_words()``.
    topic_word_scores : list of list of (word, score), length K
        Each inner list has ``num_top_words`` tuples of (word, c-TF-IDF score).
        Scores are normalized per topic to [0, 1].
    """
    scored = _compute_ctfidf_and_select(
        cluster_assignments, texts, num_top_words, mmr_diversity
    )
    top_words = [" ".join(word for word, _ in ws) for ws in scored]
    return top_words, scored
