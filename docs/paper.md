# Paper notes

## Bibliographic record

| Field | Value |
| --- | --- |
| Title | HRET: Heterogeneous Information Network for Recommendation in Testing and Inspection |
| Authors | Liwen Zhang, Weiping Li, Tong Mo, Weijie Chu |
| Affiliation | School of Software and Microelectronics, Peking University |
| Venue | 2022 International Conference on Service Science (ICSS), Zhuhai, China |
| Pages | 203–208 |
| DOI | [10.1109/ICSS55994.2022.00038](https://doi.org/10.1109/ICSS55994.2022.00038) |

## Abstract (as published)

> With the help of sufficiency of heterogeneous information, heterogeneous information
> network (HIN) has been treated as the most advanced method to extract complex semantic
> data in recommender system. But it is still an empty field for some traditional
> industries such as testing and inspection industry, which mainly adopt the
> similarity-based collaborative filtering (CF) method. It will make a huge waste of rich
> auxiliary data, which could be fully utilized by HIN based model. Especially for testing
> and inspection industry, there is a profession model to find a more accurate match
> between the user and business. In this work, we succeeded in building up an embedding
> approach for recommendation, and design a unique network structure to utilize both the
> underlying information and proper model to solve the specialty problem in testing and
> inspection industry, which is different from normal recommendation scenario. An
> intensive experiment on real world data set shows the performance of the model.

## What the paper motivates

1. **HINs are a natural fit for the domain.** Users, providers, inspection services,
   standards and regions already form a typed graph; a pure user–business matrix throws
   most of it away.
2. **Similarity-based CF is the status quo.** Traditional testing-and-inspection
   recommendation mostly compares user/business profiles, which struggles when a user has
   only a few historical interactions.
3. **The matching problem is professional, not generic.** A provider is only eligible for
   the services it is accredited for, and standards constrain the service catalogue. The
   model has to respect this structure rather than treat businesses as interchangeable
   items.

## How this repository instantiates the idea

| Paper-level idea | Concrete implementation |
| --- | --- |
| Build a HIN for testing and inspection | Five node types and six typed relations in [`hret/data/schema.py`](../hret/data/schema.py) |
| Exploit meta-path semantics | `UBU`, `USU`, `URU`, `BUB`, `BSB` (and optional `BRB`/`USBU`/`BSUB`) in [`hret/data/metapath.py`](../hret/data/metapath.py) |
| Learn embeddings on the HIN | metapath2vec with negative sampling in [`hret/models/metapath2vec.py`](../hret/models/metapath2vec.py) |
| Design a network structure over the views | Meta-path attention / concat / hybrid fusion in [`hret/models/fusion.py`](../hret/models/fusion.py) |
| Recommendation model | BPR ranking over fused user/business vectors in [`hret/models/hret.py`](../hret/models/hret.py) |
| Real-world evaluation | The protocol and baselines are provided; the paper's private dataset is not redistributed |

## Scope and reproducibility caveats

* The published paper is paywalled and its PDF is not bundled here. This repository is a
  clean-room implementation of the pipeline described in the public abstract and the HIN
  recommendation literature; it is not a line-by-line reproduction of the authors' code.
* The exact meta-path set, walk length, embedding size, negative-sampling distribution and
  training schedule are not specified in the abstract. They are exposed in
  [`configs/hret.yaml`](../configs/hret.yaml) so that they can be matched to the paper's
  experimental section.
* The bundled `data/sample` is synthetic and is only meant to exercise the pipeline. The
  numbers in the README are from that sample and are **not** the paper's results.
* If you are the author and want the repository to mirror the paper exactly, replace the
  defaults in `configs/hret.yaml` and, if needed, adjust the meta-path registry — the rest
  of the pipeline is modular.
