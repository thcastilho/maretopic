# SPDX-License-Identifier: GPL-2.0
"""
Evaluate both MARETopic variants on 20 Newsgroups under the TopMost protocol.

Downloads the dataset on first run (~20 MB into ./data), fits each variant and
reports topic coherence (C_v), topic diversity (TD), Purity and NMI.

    pip install "maretopic[topmost]"
    python examples/evaluate_20ng.py

UMAP is not seeded, so each run differs slightly; the numbers below are the
mean and standard deviation over RUNS fits, which is how the paper reports
them. Expect roughly 15 minutes per variant on a laptop CPU.
"""

from pathlib import Path

import numpy as np
from topmost import download_dataset, BasicDataset
from topmost.eva import clustering, topic_coherence, topic_diversity

from maretopic.topmost import MARETopicTrainer

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET = "20NG"
NUM_TOPICS = 50
NUM_TOP_WORDS = 15
RUNS = 3


def evaluate(top_words, test_theta, dataset):
    """C_v, TD, Purity and NMI, exactly as TopMost computes them."""
    return {
        "C_v": topic_coherence._coherence(
            dataset.train_texts, dataset.vocab, top_words,
            coherence_type="c_v", topn=NUM_TOP_WORDS,
        ),
        "TD": topic_diversity._diversity(top_words),
        **clustering._clustering(np.array(test_theta), dataset.test_labels),
    }


def run_variant(scoring, dataset):
    """Fit one variant RUNS times and return per-metric mean and std."""
    results = []
    for run in range(1, RUNS + 1):
        print(f"\n[{scoring}] run {run}/{RUNS}")
        trainer = MARETopicTrainer(
            dataset,
            num_topics=NUM_TOPICS,
            num_top_words=NUM_TOP_WORDS,
            scoring=scoring,
            cache_dir=str(CACHE_DIR / "sbert"),
            dataset_name=DATASET,
        )
        top_words, _ = trainer.train()
        _, test_theta = trainer.export_theta()
        results.append(evaluate(top_words, test_theta, dataset))

    return {
        metric: (float(np.mean([r[metric] for r in results])),
                 float(np.std([r[metric] for r in results])))
        for metric in results[0]
    }


def main():
    dataset_path = CACHE_DIR / DATASET
    if not dataset_path.exists():
        print(f"Downloading {DATASET}...")
        download_dataset(DATASET, cache_path=str(CACHE_DIR))

    dataset = BasicDataset(str(dataset_path), read_labels=True, as_tensor=True)
    print(f"{DATASET}: {len(dataset.train_texts)} train, "
          f"{len(dataset.test_texts)} test, vocab {len(dataset.vocab)}")

    scores = {
        "MARETopic_Corr": run_variant("correlation", dataset),
        "MARETopic_Diff": run_variant("diffusion", dataset),
    }

    metrics = list(next(iter(scores.values())))
    print(f"\n{DATASET} — K={NUM_TOPICS}, mean over {RUNS} runs\n")
    print(f"{'':<16}" + "".join(f"{m:>16}" for m in metrics))
    for name, values in scores.items():
        row = "".join(f"{mean:>10.4f} ±{std:.4f}" for mean, std in
                      (values[m] for m in metrics))
        print(f"{name:<16}{row}")

    # The leaders are the topics: each one is a real document in the corpus.
    trainer = MARETopicTrainer(dataset, num_topics=NUM_TOPICS,
                               cache_dir=str(CACHE_DIR / "sbert"),
                               dataset_name=DATASET)
    trainer.train()
    print("\nFirst three topics, with the document anchoring each one:\n")
    for topic, leader in zip(trainer.get_top_words()[:3],
                             trainer.model.get_leaders()[:3]):
        print(f"  words   : {topic}")
        print(f"  exemplar: {dataset.train_texts[leader][:200]}...\n")


if __name__ == "__main__":
    main()
