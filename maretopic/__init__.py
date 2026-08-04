"""
MARETopic: Manifold-Aware and Rank-based Exemplars Topic Modeling.

Training-free topic discovery over ranked lists. Each topic is anchored on a
real corpus document — a leader selected by a greedy criterion — rather than
on a latent vector or a distribution over the vocabulary.

    from maretopic import MARETopic

    model = MARETopic(num_topics=50, scoring='correlation')
    model.fit(train_texts)
    theta = model.transform(test_texts)
    top_words = model.get_top_words()
    leaders = model.get_leaders()      # indices of the exemplar documents
"""

from .model import MARETopic

__all__ = ["MARETopic"]
__version__ = "0.1.0"
