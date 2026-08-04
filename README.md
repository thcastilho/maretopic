# maretopic

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Accepted-brightgreen.svg)](#-citation)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](https://github.com/thcastilho/maretopic/blob/main/LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Official_Repo-blue?logo=github)](https://github.com/thcastilho/maretopic)

**maretopic** is the official implementation of **MARETopic** (**M**anifold-**A**ware and **R**ank-based **E**xemplars Topic Modeling), a training-free topic model that casts topic discovery as rank-based prototype selection.

> 📢 **Accepted at EMNLP 2026 (Main Conference).** If you use this package, please see the [citation](#-citation) below.

---

## 🔍 Overview

Neural topic models produce topics as latent vectors or distributions over the vocabulary — objects with no grounding in any specific text. Clustering-based pipelines pick representative documents only afterwards, by ranking cluster members against a centroid.

MARETopic makes the document the topic. A greedy algorithm selects exactly *K* **leader documents** whose neighborhoods cover the corpus, and each leader *is* the topic: a real text you can open and read.

The selection never uses absolute distances. Documents are encoded, projected onto a low-dimensional manifold, and turned into **ranked lists** — ordinal neighborhood structure, which is far less affected by the hubness and anisotropy of high-dimensional embedding spaces.

Key properties:
- **Training-free**: no gradient updates, no variational inference.
- **Exactly *K* topics**: the number of topics is an input, not an outcome of density parameters.
- **Interpretable by construction**: every topic is a corpus document.
- **Out-of-sample inference**: unseen documents are scored against the leaders without refitting.

---

## 📦 Installation

```bash
pip install maretopic
```

To also run the TopMost integration and the example:

```bash
pip install "maretopic[topmost]"
```

The latest development version can be installed directly from source:

```bash
pip install "maretopic[topmost] @ git+https://github.com/thcastilho/maretopic"
```

**Dependencies:** `numpy`, `scikit-learn`, `tqdm`, `sentence-transformers`, `umap-learn`, and [`interpretable-embeddings`](https://github.com/thcastilho/interpretable-embeddings) (which provides the canonical GRaCE and RaDE implementations).

Requires Python ≥ 3.9.

---

## 🧠 Variants

Both variants share the same greedy criterion: pick the candidate that maximizes a score while penalizing redundancy with the leaders already chosen. They differ in how those two quantities are computed, and neither is a reduced form of the other.

### MARETopic_Corr — `scoring="correlation"`
Candidates are scored by a **query performance predictor** (Reciprocal Density: how strongly a document's neighbors agree on neighborhood membership) and penalized by a **rank correlation measure** (JaccardMax: maximum Jaccard overlap across prefix depths). Either measure can be swapped without touching the rest of the pipeline.

### MARETopic_Diff — `scoring="diffusion"`
A rank-weighted affinity matrix **W** is raised to a power, `A = W²`. Its diagonal scores the candidate and its off-diagonal entries penalize redundancy, so the variant needs neither a QPP nor a correlation measure — and runs faster.

---

## 🛠 Usage

```python
from maretopic import MARETopic

model = MARETopic(
    num_topics=50,        # K
    top_K=100,            # ranked-list depth
    umap_dim=5,           # 0 disables the UMAP projection
    mmr_diversity=0.3,    # inter-topic MMR weight for keyword extraction
    scoring="correlation" # or "diffusion"
)

model.fit(train_texts)

for words in model.get_top_words():
    print(words)
```

Each topic is anchored on a real document, and `get_leaders()` returns their indices in the training corpus:

```python
for topic_id, leader in enumerate(model.get_leaders()):
    print(f"topic {topic_id} is the document: {train_texts[leader][:120]}...")
```

Unseen documents are projected with the fitted reducer and scored against the leaders — no refitting:

```python
theta = model.transform(test_texts)   # (n_documents, num_topics)
assignments = theta.argmax(axis=1)
```

### Reproducibility

UMAP runs without a fixed `random_state`, so results vary slightly between fits — the same behavior as BERTopic. Report the mean and standard deviation over several runs, as the paper does. Passing `umap_random_state` fixes the seed at the cost of UMAP's parallelism.

---

## 🔗 TopMost integration

`maretopic.topmost.MARETopicTrainer` implements the same trainer interface as TopMost's own `BERTopicTrainer` and `FASTopicTrainer`, so MARETopic drops into a [TopMost](https://github.com/BobXWu/TopMost) evaluation pipeline. **Nothing in your `topmost` installation needs to be patched.**

```python
from topmost import BasicDataset, download_dataset
from topmost.eva import clustering, topic_coherence, topic_diversity
from maretopic.topmost import MARETopicTrainer

download_dataset("20NG", cache_path="./data")
dataset = BasicDataset("./data/20NG", read_labels=True, as_tensor=True)

trainer = MARETopicTrainer(dataset, num_topics=50, num_top_words=15)
top_words, train_theta = trainer.train()
train_theta, test_theta = trainer.export_theta()
```

A complete, runnable version — both variants, all four metrics — is in [`examples/evaluate_20ng.py`](examples/evaluate_20ng.py).

---

## 📁 Package Structure

```
maretopic/
│
├── model.py         # MARETopic class: fit, transform, get_top_words, get_leaders
├── embeddings.py    # SBERT encoding + ranked-list construction (BallTree)
├── words.py         # c-TF-IDF + inter-topic MMR keyword extraction
└── topmost.py       # MARETopicTrainer adapter for the TopMost toolkit
```

Leader selection and the training document–topic matrix are delegated to `interpretable-embeddings` (`GRaCE` for the correlation variant, `RaDE` for the diffusion variant), so the algorithms have a single canonical implementation.

---

## 📚 Citation

If you use this package in your research, please cite:

> **Almeida, T. C. C., Pedronette, D. C. G.**
> *A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes*
> Accepted at the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP), Main Conference.

```bibtex
@inproceedings{almeida2026maretopic,
  title     = {A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes},
  author    = {Almeida, Thiago C. C. and Pedronette, Daniel C. G.},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

MARETopic builds directly on two earlier methods from the same line of work:

> **Almeida, T. C. C., Letício, G. R., Valem, L. P., Freitas, A., Pedronette, D. C. G.**
> *Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data*
> 2025 International Joint Conference on Neural Networks (IJCNN), Rome – Italy.
> [![GRaCE](https://img.shields.io/badge/View%20Paper-GRaCE-blue)](https://doi.org/10.1109/IJCNN64981.2025.11229362)

> **De Fernando, F. A., Pedronette, D. C. G., De Sousa, G. J., Valem, L. P., Guilherme, I. R.**
> *RaDE+: A semantic rank-based graph embedding algorithm*
> International Journal of Information Management Data Insights, 2(2), 100078, 2022.
> [![RaDE+](https://img.shields.io/badge/View%20Paper-RaDE+-blue)](https://doi.org/10.1016/j.jjimei.2022.100078)

---

## 🤝 Contact

- Thiago César Castilho Almeida: `tc.almeida@unesp.br`
- Daniel Carlos Guimarães Pedronette: `daniel.pedronette@unesp.br`
