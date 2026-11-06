"""metapath2vec pre-training for HRET.

For every meta-path we generate typed random walks, split each walk into
per-node-type sequences, and train a skip-gram model with negative sampling
on those sequences. The resulting vectors are used to initialise the
meta-path specific embedding tables of :class:`~hret.models.hret.HRET`.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..data.metapath import MetaPath, generate_walks, skipgram_pairs, type_sequences
from ..data.schema import HIN


class SkipGram(nn.Module):
    """Skip-gram with negative sampling (word2vec style)."""

    def __init__(self, num_nodes: int, dim: int) -> None:
        super().__init__()
        self.center = nn.Embedding(num_nodes, dim)
        self.context = nn.Embedding(num_nodes, dim)
        bound = 0.5 / dim
        nn.init.uniform_(self.center.weight, -bound, bound)
        nn.init.zeros_(self.context.weight)

    def loss(
        self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        center = self.center(centers)
        context = self.context(contexts)
        positive = (center * context).sum(dim=-1)
        negative = torch.bmm(self.context(negatives), center.unsqueeze(-1)).squeeze(-1)
        return (-F.logsigmoid(positive) - F.logsigmoid(-negative).sum(dim=-1)).mean()


@dataclass
class MetaPathEmbedding:
    """Pre-trained vectors for one meta-path."""

    path: str
    embeddings: Dict[str, np.ndarray]
    walks: int
    pairs: int


def _train_skipgram(
    model: SkipGram,
    pairs: Sequence[Tuple[int, int]],
    num_nodes: int,
    cfg: ModelConfig,
    device: torch.device,
    seed: int,
) -> float:
    if not pairs:
        return 0.0

    data = np.asarray(pairs, dtype=np.int64)
    generator = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    model.train()

    last_loss = 0.0
    for _epoch in range(cfg.pretrain_epochs):
        order = generator.permutation(len(data))
        for start in range(0, len(data), cfg.pretrain_batch_size):
            batch = order[start : start + cfg.pretrain_batch_size]
            if len(batch) == 0:
                continue
            centers = torch.from_numpy(data[batch, 0]).to(device)
            contexts = torch.from_numpy(data[batch, 1]).to(device)
            negatives = torch.from_numpy(
                generator.integers(0, num_nodes, size=(len(batch), cfg.neg_samples))
            ).to(device)

            optimizer.zero_grad()
            loss = model.loss(centers, contexts, negatives)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    return last_loss


def pretrain_meta_path(
    hin: HIN,
    meta_path: MetaPath,
    cfg: ModelConfig,
    device: torch.device | str = "cpu",
    seed: int = 0,
    logger: logging.Logger | None = None,
) -> MetaPathEmbedding:
    """Pre-train meta-path specific embeddings for every node type on the path."""
    device = torch.device(device)
    rng = random.Random(seed)
    walks = generate_walks(
        hin,
        meta_path,
        walks_per_node=cfg.walks_per_node,
        walk_length=cfg.walk_length,
        rng=rng,
    )
    sequences = type_sequences(walks, meta_path)

    embeddings: Dict[str, np.ndarray] = {}
    total_pairs = 0
    for node_type, seqs in sequences.items():
        pairs = skipgram_pairs(seqs, cfg.window_size)
        total_pairs += len(pairs)
        model = SkipGram(hin.num_nodes[node_type], cfg.dim).to(device)
        loss = _train_skipgram(
            model,
            pairs,
            num_nodes=hin.num_nodes[node_type],
            cfg=cfg,
            device=device,
            seed=seed + 1,
        )
        embeddings[node_type] = model.center.weight.detach().cpu().numpy().copy()
        if logger is not None:
            logger.info(
                "metapath2vec %-4s | %-8s | walks=%d pairs=%d final_loss=%.4f",
                meta_path.name,
                node_type,
                len(walks),
                len(pairs),
                loss,
            )

    return MetaPathEmbedding(
        path=meta_path.name,
        embeddings=embeddings,
        walks=len(walks),
        pairs=total_pairs,
    )


def pretrain_all(
    hin: HIN,
    meta_paths: Sequence[MetaPath],
    cfg: ModelConfig,
    device: torch.device | str = "cpu",
    seed: int = 0,
    logger: logging.Logger | None = None,
) -> Dict[str, MetaPathEmbedding]:
    """Pre-train every configured meta-path."""
    results: Dict[str, MetaPathEmbedding] = {}
    for index, meta_path in enumerate(meta_paths):
        results[meta_path.name] = pretrain_meta_path(
            hin,
            meta_path,
            cfg,
            device=device,
            seed=seed + 1000 * (index + 1),
            logger=logger,
        )
    return results
