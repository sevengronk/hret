"""BPR training loop shared by HRET and the matrix-factorisation baseline."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ..config import TrainConfig
from ..evaluation import EvalResult


@dataclass
class TrainingHistory:
    epochs: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_metrics: Dict[str, float] = field(default_factory=dict)


def _sample_negatives(
    users: Sequence[int],
    num_items: int,
    positives: Mapping[int, Sequence[int]],
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    negatives = generator.integers(0, num_items, size=(len(users), count))
    for row, user in enumerate(users):
        forbidden = positives.get(user, ())
        if not forbidden:
            continue
        for column in range(count):
            attempts = 0
            while negatives[row, column] in forbidden and attempts < 30:
                negatives[row, column] = int(generator.integers(0, num_items))
                attempts += 1
    return negatives


def train_bpr(
    model: torch.nn.Module,
    train_by_user: Mapping[int, Sequence[int]],
    num_items: int,
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
    val_fn: Optional[Callable[[torch.nn.Module], EvalResult]] = None,
    monitor: str = "NDCG@10",
    logger: Optional[logging.Logger] = None,
    seed: int = 0,
) -> TrainingHistory:
    """Train ``model`` with BPR and optional early stopping on ``monitor``."""
    device = torch.device(device)
    model.to(device)
    model.train()

    users = [user for user, items in train_by_user.items() for _ in items]
    positives = [item for items in train_by_user.values() for item in items]
    if not users:
        raise ValueError("No training interactions")
    pairs = np.column_stack(
        [np.asarray(users, dtype=np.int64), np.asarray(positives, dtype=np.int64)]
    )
    positives_set = {user: set(items) for user, items in train_by_user.items()}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    generator = np.random.default_rng(seed)
    history = TrainingHistory()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -float("inf")
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        order = generator.permutation(len(pairs))
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(pairs), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            batch_users = pairs[batch, 0]
            batch_positive = pairs[batch, 1]
            negatives = _sample_negatives(
                batch_users.tolist(), num_items, positives_set, 1, generator
            )[:, 0]

            user_ids = torch.from_numpy(batch_users).to(device)
            positive_ids = torch.from_numpy(batch_positive).to(device)
            negative_ids = torch.from_numpy(negatives).to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(user_ids, positive_ids, negative_ids)
            if cfg.l2_reg > 0 and hasattr(model, "embedding_l2"):
                loss = loss + cfg.l2_reg * model.embedding_l2()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            num_batches += 1

        epoch_record = {
            "epoch": float(epoch),
            "loss": total_loss / max(1, num_batches),
            "seconds": time.time() - started,
        }

        if val_fn is not None and epoch % max(1, cfg.eval_every) == 0:
            model.eval()
            result = val_fn(model)
            model.train()
            epoch_record.update(result.metrics)
            score = result.metrics.get(monitor, result.metrics.get("MRR", 0.0))
            if score > best_score + 1e-9:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                history.best_epoch = epoch
                history.best_metrics = dict(result.metrics)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if logger is not None:
                logger.info(
                    "epoch %3d | loss %.4f | %s %.4f | best %.4f @ %d",
                    epoch,
                    epoch_record["loss"],
                    monitor,
                    score,
                    best_score,
                    history.best_epoch,
                )
        elif logger is not None:
            logger.info("epoch %3d | loss %.4f | %.1fs", epoch, epoch_record["loss"], epoch_record["seconds"])

        history.epochs.append(epoch_record)
        if val_fn is not None and stale_epochs >= cfg.patience:
            if logger is not None:
                logger.info("Early stopping after %d stale epochs", stale_epochs)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
