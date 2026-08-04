# SPDX-License-Identifier: GPL-2.0
# MARETopic — model.py
# Main class: the full topic modeling pipeline over ranked lists.

from __future__ import annotations

import math
import time
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree
from tqdm import tqdm

# Canonical implementations from interpretable-embeddings. GRaCE/RaDE drive
# train-side leader selection + embedding (delegated in fit()); compute_jacmax
# backs the out-of-sample transform() correlation path.
from interpretable_embeddings.grace import GRaCE
from interpretable_embeddings.rade import RaDE
from interpretable_embeddings.measures.correlation import compute_jacmax

from .embeddings import encode_sbert, build_ranked_lists
from .words import extract_topic_words_with_scores


def _query_oos_knn(test_emb: np.ndarray, train_emb: np.ndarray, k: int) -> np.ndarray:
    """k-NN of test docs against train corpus. Returns (N_test, k) int64 array."""
    tr_norms = np.linalg.norm(train_emb, axis=1, keepdims=True)
    tr_norms = np.where(tr_norms == 0, 1.0, tr_norms)
    te_norms = np.linalg.norm(test_emb, axis=1, keepdims=True)
    te_norms = np.where(te_norms == 0, 1.0, te_norms)

    train_normed = (train_emb / tr_norms).astype(np.float64)
    test_normed = (test_emb / te_norms).astype(np.float64)
    tree = BallTree(train_normed)
    _, indices = tree.query(test_normed, k=k)
    return indices


class MARETopic:
    """
    MARETopic: Manifold-Aware and Rank-based Exemplars Topic Modeling.

    A training-free topic model based on rank-based prototype selection. Each
    topic is represented by a real document (exemplar) chosen by a greedy
    algorithm operating on ranked lists built from dense semantic embeddings.

    Two variants share that criterion, selected by ``scoring``. Neither is a
    reduced version of the other — they differ in how a candidate is scored
    and how redundancy with already-selected leaders is penalised.

    ``'correlation'`` (**MARETopic_Corr**, default)
        Candidates are scored by a query performance predictor (reciprocal
        density) and penalised by a rank correlation measure (JaccardMax).

    ``'diffusion'`` (**MARETopic_Diff**)
        Both roles are played by the diffusion matrix A = W²: the diagonal
        scores the candidate, the off-diagonal penalises redundancy. Needs
        neither measure, and runs faster.

    Parameters
    ----------
    num_topics : int
        Number of topics K (leaders) to select.
    top_K : int
        Ranked-list depth — neighbourhood size for similarity and correlation.
    umap_dim : int
        UMAP output dimensionality applied before building ranked lists.
        Set to 0 to skip UMAP (use raw SBERT embeddings).
    mmr_diversity : float
        Inter-topic MMR diversity weight for keyword extraction.
        0 = no deduplication; 0.3 = recommended sweet spot.
    scoring : str
        ``'correlation'`` (MARETopic_Corr) or ``'diffusion'`` (MARETopic_Diff).
    sbert_model : str
        Sentence-Transformers model name.
    cache_dir : path-like, optional
        Directory for SBERT embedding cache. UMAP is not cached — the fitted
        reducer is stored as model state and reused in ``transform()``.
        Note: UMAP has no fixed random_state, so results vary across runs
        (same as BERTopic). Report mean ± std over 3 runs in experiments.
    dataset_name : str, optional
        Identifier used for cache file naming (e.g. ``'20NG'``).

    Notes
    -----
    Leader selection and the training document-topic matrix come from the
    ``interpretable-embeddings`` package — ``GRaCE`` for ``'correlation'``,
    ``RaDE`` for ``'diffusion'`` — so the algorithms have a single canonical
    implementation. What lives here is the text pipeline around them and the
    out-of-sample inference in ``transform()``, which those classes do not
    provide.
    """

    def __init__(
        self,
        num_topics: int = 50,
        top_K: int = 100,
        umap_dim: int = 5,
        mmr_diversity: float = 0.3,
        scoring: str = "correlation",
        sbert_model: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path | None = None,
        dataset_name: str | None = None,
        umap_random_state: int | None = None,
    ) -> None:
        if scoring not in ("correlation", "diffusion"):
            raise ValueError(
                f"scoring must be 'correlation' or 'diffusion', got {scoring!r}"
            )

        self.num_topics = num_topics
        self.top_K = top_K
        self.umap_dim = umap_dim
        self.mmr_diversity = mmr_diversity
        self.scoring = scoring
        self.sbert_model = sbert_model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.dataset_name = dataset_name or "dataset"
        self.umap_random_state = umap_random_state

        # State set after fit()
        self._train_emb: np.ndarray | None = None   # reduced train embeddings
        self._train_rks: list[list[int]] | None = None
        self._leaders: list[int] | None = None
        self._train_theta: np.ndarray | None = None
        self._top_words: list[str] | None = None
        self._top_words_scored: list[list[tuple[str, float]]] | None = None
        self._umap_reducer = None                    # fitted umap.UMAP instance

        # Stage-level wall-clock times populated by fit() and transform().
        # Keys: sbert_encode, umap, index, selection, words, transform.
        # All values in seconds; absent if that stage was not executed.
        self.stage_times_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _sbert_cache(self, split: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_model = self.sbert_model.replace("/", "_")
        return self.cache_dir / f"{self.dataset_name}_{split}_{safe_model}.npy"

    # ------------------------------------------------------------------
    # Out-of-sample (transform) internals.
    # Train-side selection + embedding are delegated to GRaCE/RaDE in fit();
    # only the out-of-sample inference below is MARETopic-specific.
    # ------------------------------------------------------------------

    def _compute_test_theta_correlation(
        self,
        test_emb: np.ndarray,
        train_emb: np.ndarray,
        rks: list[list[int]],
        leaders: list[int],
    ) -> np.ndarray:
        """Out-of-sample theta: build test ranked lists against train corpus."""
        N_test = len(test_emb)
        K = len(leaders)
        k = min(self.top_K + 1, len(train_emb))

        indices = _query_oos_knn(test_emb, train_emb, k)  # (N_test, k)

        print("  [MARETopic] Computing test θ (out-of-sample JacMax)...")
        theta = np.zeros((N_test, K), dtype=np.float32)
        for i in tqdm(range(N_test)):
            test_rk = indices[i].tolist()           # already excludes nothing — train BallTree
            for j, ld in enumerate(leaders):
                theta[i, j] = compute_jacmax(
                    test_rk, rks[ld][: self.top_K], self.top_K
                )

        return theta

    def _compute_test_theta_diffusion(
        self,
        test_emb: np.ndarray,
        train_emb: np.ndarray,
        leaders: list[int],
    ) -> np.ndarray:
        """Out-of-sample theta for diffusion mode: rank-based 1-hop affinity."""
        N_test = len(test_emb)
        N_train = len(train_emb)
        K = len(leaders)
        top_L = self.top_K
        log_L = math.log(top_L)

        # Query top_L neighbours for each test doc to build position map
        k = min(top_L, N_train)
        indices = _query_oos_knn(test_emb, train_emb, k)  # (N_test, k)

        print("  [MARETopic] Computing test θ (rank-based affinity)...")
        theta = np.zeros((N_test, K), dtype=np.float32)

        for i in tqdm(range(N_test)):
            # Build rank map: train_idx → position in test doc's ranked list
            rank_map = {int(idx): pos for pos, idx in enumerate(indices[i])}
            for j, ld in enumerate(leaders):
                pos = rank_map.get(ld, top_L)   # default: not in top_L → weight 0
                if pos < top_L:
                    theta[i, j] = 1.0 - math.log(pos + 1) / log_L

        return theta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_texts: list[str], num_top_words: int = 15) -> "MARETopic":
        """
        Fit MARETopic on training documents.

        Parameters
        ----------
        train_texts : list of str
            Training corpus (preprocessed documents).
        num_top_words : int
            Keywords per topic to extract and store for ``get_top_words()``.

        Returns
        -------
        self
        """
        self.stage_times_ = {}

        print(f"[MARETopic] Encoding {len(train_texts)} documents (SBERT)...")
        _t = time.perf_counter()
        train_emb = encode_sbert(
            train_texts,
            model_name=self.sbert_model,
            cache_path=self._sbert_cache("train"),
        )
        self.stage_times_["sbert_encode"] = time.perf_counter() - _t

        if self.umap_dim > 0:
            print(
                f"[MARETopic] UMAP: {train_emb.shape[1]}d → {self.umap_dim}d "
                f"(n_neighbors=15, min_dist=0.0, cosine)..."
            )
            import umap as umap_lib
            _t = time.perf_counter()
            self._umap_reducer = umap_lib.UMAP(
                n_components=self.umap_dim,
                n_neighbors=15,
                min_dist=0.0,
                metric="cosine",
                random_state=self.umap_random_state,
            )
            train_emb = self._umap_reducer.fit_transform(train_emb).astype(np.float32)
            self.stage_times_["umap"] = time.perf_counter() - _t

        print(f"[MARETopic] Building ranked lists (top_K={self.top_K})...")
        _t = time.perf_counter()
        rks = build_ranked_lists(train_emb, self.top_K)
        self.stage_times_["index"] = time.perf_counter() - _t

        # Train-side leader selection + embedding are delegated to the canonical
        # interpretable-embeddings classes (GRaCE for correlation, RaDE for
        # diffusion), so this package never carries a second copy of the
        # algorithms. Out-of-sample inference (transform) has no equivalent in
        # those classes and stays inline.
        _t = time.perf_counter()
        if self.scoring == "correlation":
            grace = GRaCE(rks=rks, top_K=self.top_K,
                          correlation_measure="jacmax",
                          estimation_measure="reciprocal_density")
            grace.compute_estimations()
            grace.compute_leaders(num_leaders=self.num_topics)
            leaders = grace.get_leaders()
            train_theta = grace.transform().astype(np.float32)
        else:  # diffusion
            # num_candidates = N removes RaDE's candidate restriction so the
            # greedy selection ranges over the full pool, matching MARETopic_Diff.
            rade = RaDE(rks=rks, rks_size_L=self.top_K)
            rade.compute_W_matrix()
            rade.compute_leaders(num_candidates=len(rks),
                                 num_leaders=self.num_topics, t=2)
            leaders = rade.get_leaders()
            train_theta = rade.transform().astype(np.float32)
        self.stage_times_["selection"] = time.perf_counter() - _t

        print(f"[MARETopic] Extracting topic words (MMR diversity={self.mmr_diversity})...")
        _t = time.perf_counter()
        train_clusters = np.argmax(train_theta, axis=1)
        top_words, top_words_scored = extract_topic_words_with_scores(
            train_clusters, train_texts, num_top_words, self.mmr_diversity
        )
        self.stage_times_["words"] = time.perf_counter() - _t

        # Store model state
        self._train_emb = train_emb
        self._train_rks = rks
        self._leaders = leaders
        self._train_theta = train_theta
        self._top_words = top_words
        self._top_words_scored = top_words_scored

        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Compute topic distributions for new (out-of-sample) documents.

        Parameters
        ----------
        texts : list of str
            Documents to encode and assign to topics.

        Returns
        -------
        theta : np.ndarray of shape (N, num_topics), dtype float32
        """
        if self._leaders is None:
            raise RuntimeError("Call fit() before transform().")

        print(f"[MARETopic] Encoding {len(texts)} documents (SBERT)...")
        _t = time.perf_counter()
        emb = encode_sbert(texts, model_name=self.sbert_model)
        self.stage_times_["transform_sbert"] = time.perf_counter() - _t

        if self.umap_dim > 0:
            print(f"[MARETopic] UMAP projection ({emb.shape[1]}d → {self.umap_dim}d)...")
            _t = time.perf_counter()
            emb = self._umap_reducer.transform(emb).astype(np.float32)
            self.stage_times_["transform_umap"] = time.perf_counter() - _t

        _t = time.perf_counter()
        if self.scoring == "correlation":
            result = self._compute_test_theta_correlation(
                emb, self._train_emb, self._train_rks, self._leaders
            )
        else:
            result = self._compute_test_theta_diffusion(
                emb, self._train_emb, self._leaders
            )
        self.stage_times_["transform_index"] = time.perf_counter() - _t
        return result

    def fit_transform(
        self, train_texts: list[str], num_top_words: int = 15
    ) -> np.ndarray:
        """Fit on train_texts and return train theta."""
        self.fit(train_texts, num_top_words=num_top_words)
        return self._train_theta

    def get_top_words(self) -> list[str]:
        """
        Return topic keywords extracted during ``fit()``.

        Returns
        -------
        top_words : list of str, length num_topics
            Each element is a space-separated keyword string.
        """
        if self._top_words is None:
            raise RuntimeError("Call fit() before get_top_words().")
        return self._top_words

    def get_leaders(self) -> list[int]:
        """Return indices (in the training corpus) of the selected leader documents."""
        if self._leaders is None:
            raise RuntimeError("Call fit() before get_leaders().")
        return self._leaders

    def get_top_words_with_scores(self) -> list[list[tuple[str, float]]]:
        """
        Return topic keywords with their normalized c-TF-IDF scores.

        Returns
        -------
        topic_word_scores : list of list of (word, score), length num_topics
            Each inner list has ``num_top_words`` tuples of (word, c-TF-IDF).
            Scores are normalized per topic to [0, 1].
        """
        if self._top_words_scored is None:
            raise RuntimeError("Call fit() before get_top_words_with_scores().")
        return self._top_words_scored
