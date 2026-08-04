# SPDX-License-Identifier: GPL-2.0
# MARETopic — topmost.py
# Adapter that plugs MARETopic into the TopMost evaluation toolkit.

from __future__ import annotations

from .model import MARETopic


class MARETopicTrainer:
    """
    TopMost trainer wrapper for MARETopic.

    Duck-types the same interface as TopMost's own ``BERTopicTrainer`` and
    ``FASTopicTrainer`` — ``train()``, ``test()``, ``get_top_words()``,
    ``export_theta()`` — so it drops into a TopMost evaluation pipeline
    without patching the installed ``topmost`` package.

    Parameters
    ----------
    dataset : TopMost BasicDataset
    num_topics : int
        Number of topics K.
    num_top_words : int
        Keywords per topic.
    scoring : str
        ``'correlation'`` (MARETopic_Corr, default) or ``'diffusion'``
        (MARETopic_Diff).
    top_K : int
        Ranked-list depth.
    umap_dim : int
        UMAP output dimensionality (0 = skip UMAP).
    mmr_diversity : float
        MMR inter-topic diversity weight.
    sbert_model : str
        Sentence-Transformers model name.
    cache_dir : str, optional
        Directory for SBERT embedding cache.
    dataset_name : str, optional
        Identifier used in cache file names (e.g. ``'20NG'``).
    """

    def __init__(
        self,
        dataset,
        num_topics: int = 50,
        num_top_words: int = 15,
        scoring: str = "correlation",
        top_K: int = 100,
        umap_dim: int = 5,
        mmr_diversity: float = 0.3,
        sbert_model: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.model = MARETopic(
            num_topics=num_topics,
            top_K=top_K,
            umap_dim=umap_dim,
            mmr_diversity=mmr_diversity,
            scoring=scoring,
            sbert_model=sbert_model,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
        )

    def train(self):
        """
        Fit MARETopic on the training corpus.

        Returns
        -------
        top_words : list of str
        train_theta : np.ndarray of shape (N_train, num_topics)
        """
        self.model.fit(self.dataset.train_texts, num_top_words=self.num_top_words)
        return self.get_top_words(), self.model._train_theta

    def test(self, texts) -> "np.ndarray":
        """
        Compute topic distributions for a list of texts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        theta : np.ndarray of shape (N, num_topics)
        """
        return self.model.transform(texts)

    def get_top_words(self, num_top_words: int | None = None) -> list[str]:
        """Return topic keywords (extracted during training)."""
        return self.model.get_top_words()

    def export_theta(self):
        """
        Return topic distributions for the full train and test splits.

        Returns
        -------
        train_theta : np.ndarray of shape (N_train, num_topics)
        test_theta  : np.ndarray of shape (N_test,  num_topics)
        """
        train_theta = self.model._train_theta
        test_theta = self.test(self.dataset.test_texts)
        return train_theta, test_theta
