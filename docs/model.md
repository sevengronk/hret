# HRET model notes

This document expands the method section of the README and records the engineering
choices behind the implementation. It is meant to be read together with
[`hret/models/hret.py`](../hret/models/hret.py),
[`hret/models/fusion.py`](../hret/models/fusion.py) and
[`hret/models/metapath2vec.py`](../hret/models/metapath2vec.py).

## 1. Problem setting

We are given a set of users $U$, businesses $B$, services $S$, standards $T$ and regions
$R$. An implicit interaction $(u, b, s, t)$ means that user $u$ consulted business $b$
for service $s$ at time $t$. The task is, for every user, to rank businesses that the
user has not consulted before.

Testing and inspection differs from movie or product recommendation in three ways:

1. **The catalogue is accreditation-driven.** A business can only be recommended for the
   services it is accredited for, and services are tied to standards.
2. **Location matters.** Inspection is often procured locally or within a region.
3. **Interactions are sparse.** A small enterprise may only ever request a handful of
   tests, so collaborative filtering alone is weak.

The HIN is the mechanism that injects 1–3 into the representation learning.

## 2. HIN construction

Nodes and edges follow [`hret/data/schema.py`](../hret/data/schema.py):

```text
user --interacts_with--> business
user --demands---------> service
business --provides----> service
user --located_in------> region
business --located_in--> region
service --governed_by--> standard
```

Only **training** interactions are used to create `user–business` and `user–service`
edges. Held-out validation and test interactions never enter the graph. Static edges
(`business–service`, `user–region`, `business–region`, `service–standard`) are available
for all nodes, which is what makes cold-start users reachable at all.

## 3. Meta-path guided random walks

A meta-path is a type sequence $P = (A_0, A_1, \ldots, A_L)$ with $A_0 = A_L$. For a
symmetric path of length $L$, the type of position $i$ is

$$
\phi(i) = A_{i \bmod L}.
$$

The walk starts at a node of type $A_0$, and at step $i$ samples uniformly from the
neighbours of the current node whose type is $\phi(i)$. If a node has no neighbour of
the required type the walk stops early. Walks shorter than three nodes are discarded.

This is the standard metapath2vec construction, with one implementation detail worth
calling out: the schema is repeated modulo $L$ so that `walk_length` can be longer than
the meta-path itself. For `UBU` the types are `user, business, user, business, ...`.

## 4. metapath2vec pre-training

For each meta-path and each node type on the path, a typed walk is split into a sequence
of same-type nodes. For example

```text
walk:  u0 b3 u7 b1 u2
user  sequence: u0, u7, u2
business sequence: b3, b1
```

A skip-gram model with negative sampling is trained on every sequence:

$$
\mathcal{L}_{\text{SG}} = - \sum_{(c, o) \in \mathcal{D}}
\log \sigma(\mathbf{e}_o^\top \mathbf{e}_c)
- \sum_{k=1}^{K} \log \sigma(-\mathbf{e}_{n_k}^\top \mathbf{e}_c),
$$

where $c$ is the centre node, $o$ a context node inside the window, and $n_k$ uniformly
sampled negatives. The centre vectors are kept as the meta-path specific embedding of a
node.

This produces one embedding table per `(meta-path, node type)`, e.g.
`(UBU, user)` and `(UBU, business)`. The tables are independent across meta-paths so that
each view can specialise.

## 5. Meta-path attention fusion

A user appears in `UBU`, `USU` and `URU`; a business appears in `BUB` and `BSB`.
Let $e_i \in \mathbb{R}^d$ be the view of a node under meta-path $i$. HRET computes

$$
\begin{aligned}
\bar{e} &= \frac{1}{P}\sum_{i=1}^{P} e_i, \\
q &= W_q \bar{e}, \\
\alpha_i &= \frac{\exp(q^\top W_k e_i / \sqrt{d})}
                 {\sum_{j=1}^{P}\exp(q^\top W_k e_j / \sqrt{d})}, \\
h &= \operatorname{LayerNorm}\left(\sum_{i=1}^{P}\alpha_i W_v e_i + \bar{e}\right).
\end{aligned}
$$

The residual connection keeps the fused vector close to the average view in the
beginning of training, while the attention can down-weight a meta-path that is noisy for
a particular node. `model.fusion: mean` disables attention and is useful as an ablation.

The attention weights are returned by `MetaPathFusion.forward` and can be inspected for
interpretability, e.g. whether a user is driven more by behaviour (`UBU`) or by demand
(`USU`). `model.fusion` also accepts `concat` (concatenate all views and project),
`hybrid` (concat plus attention) and `mean` (uniform average) for ablation studies.

## 6. Scoring and objective

Two scorers are implemented:

```text
dot:  s(u, b) = h_uᵀ h_b
mlp:  s(u, b) = MLP([h_u ‖ h_b ‖ h_u ⊙ h_b])
```

The model is trained with BPR:

$$
\mathcal{L} = -\frac{1}{|\mathcal{D}|}\sum_{(u,b^+)\in\mathcal{D}}
\log \sigma\left(s(u,b^+) - s(u,b^-)\right) + \lambda \|\Theta\|_2^2,
$$

where $b^-$ is sampled uniformly from businesses the user has not interacted with in the
training set. Embedding L2 is divided by the vocabulary size so that `train.l2_reg` is
comparable across datasets. Gradients are clipped at `train.grad_clip`.

## 7. Training algorithm

```text
1. Load data, split each user's history (train / val / test).
2. Build the HIN from training interactions + static schema edges.
3. For each meta-path:
     a. sample typed random walks,
     b. split into per-type sequences,
     c. train skip-gram and keep the centre embeddings.
4. Initialise HRET's per-path embedding tables with step 3.
5. For each epoch:
     a. sample (user, positive, negative) triples,
     b. minimise BPR + L2,
     c. evaluate validation NDCG@K, keep the best state,
     d. stop after `train.patience` epochs without improvement.
6. Evaluate the best state on the test split; export top-K lists.
```

## 8. Complexity

Let $N$ be the number of nodes, $E$ the number of edges, $P$ the number of meta-paths,
$W$ the walks per node, $L$ the walk length, and $d$ the embedding size.

* Walk generation: $O(P \cdot N \cdot W \cdot L)$ neighbour lookups.
* Skip-gram: linear in the number of context pairs, $O(P \cdot N \cdot W \cdot L \cdot K \cdot d)$
  per epoch.
* Fusion and scoring: $O(P \cdot d)$ per node; the MLP scorer costs
  $O(d \cdot h)$ per pair.

The bundled sample (240 users, 80 businesses) pre-trains in under 10 seconds on a laptop
CPU and trains in about a second.

## 9. Deviations and reproducibility notes

The paper's exact meta-path set, embedding size, walk length and negative-sampling
distribution are experiment-specific. This repository exposes them in
[`configs/hret.yaml`](../configs/hret.yaml); the defaults are a sensible reference
configuration for the schema above, not a claim that they reproduce the paper's table.
When adapting the code to a new dataset, start from the defaults and tune
`model.dim`, `model.walks_per_node`, `model.walk_length` and `train.l2_reg`.

Two further implementation choices are deliberate:

* **Businesses, not visits, are ranked.** Repeated interactions with the same business
  are collapsed before splitting, otherwise the "next item" could be a business the user
  already knows.
* **The graph is leakage-free.** Validation/test edges are excluded from the training
  HIN. This can make the task harder than a transductive setup, but it is the honest
  protocol.

## 10. Reference run on the synthetic sample

The numbers below come from the **synthetic** `data/sample` dataset, not from the paper's
real-world data. They are the mean ± population std over seeds 42, 7 and 2024. They are
useful to check that the pipeline runs and that HRET beats the baselines on a controlled
signal; they must not be compared with the paper's table.

```bash
python scripts/benchmark.py --config configs/hret.yaml --output runs/benchmark \
    --seeds 42 7 2024
```

| Model | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 10.28 ± 1.71 | 18.06 ± 0.52 | 5.86 ± 1.12 | 8.35 ± 0.72 | 8.24 ± 1.00 |
| BPR-MF | 9.03 ± 3.61 | 17.78 ± 3.94 | 5.23 ± 1.96 | 8.01 ± 2.05 | 7.96 ± 1.35 |
| **HRET** | **12.08 ± 1.80** | **23.61 ± 2.39** | **7.41 ± 0.65** | **11.14 ± 0.68** | **10.41 ± 0.23** |

Percentages on the test split, bundled synthetic sample. These are **not** the paper's
results.

HRET improves over both baselines on this controlled sample, which is the expected
sanity check: the sample contains a category/region preference that the meta-path views
can recover. The absolute numbers are small because the catalogue is small and the
held-out target is only one business per user.
