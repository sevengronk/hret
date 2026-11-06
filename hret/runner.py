"""End-to-end experiment runner: data -> pre-training -> HRET -> evaluation."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .baselines import BPRMF, PopularityRecommender, interaction_counts
from .config import Config
from .data.dataset import Dataset, Split, build_hin, leave_one_out_split, load_dataset
from .data.metapath import MetaPath
from .data.schema import HIN
from .evaluation import EvalResult, evaluate_ranking, format_metrics
from .models.hret import HRET
from .models.metapath2vec import MetaPathEmbedding, pretrain_all
from .training.trainer import TrainingHistory, train_bpr
from .utils import get_logger, save_json, set_seed


@dataclass
class Bundle:
    """Everything needed to train or evaluate HRET."""

    dataset: Dataset
    split: Split
    hin: HIN
    meta_paths: List[MetaPath]
    user_paths: List[str]
    business_paths: List[str]
    pretrained: Optional[Dict[str, MetaPathEmbedding]] = None


def prepare_bundle(
    cfg: Config,
    device: str | torch.device = "cpu",
    logger: Optional[logging.Logger] = None,
    pretrain: bool = True,
) -> Bundle:
    """Load data, build the training HIN and optionally pre-train embeddings."""
    dataset = load_dataset(cfg.data.dir)
    split = leave_one_out_split(
        dataset,
        cfg.data.min_interactions,
        strategy=cfg.data.split_strategy,
        seed=cfg.seed,
    )
    hin = build_hin(dataset, split.train)
    meta_paths = MetaPath.parse_many(cfg.model.meta_paths)

    user_paths = [path.name for path in meta_paths if path.center_type == "user"]
    business_paths = [path.name for path in meta_paths if path.center_type == "business"]
    if not user_paths or not business_paths:
        raise ValueError(
            "The configured meta-paths must cover both users and businesses; "
            f"got user={user_paths}, business={business_paths}"
        )

    if logger is not None:
        logger.info("dataset: %s", dataset.summary())
        logger.info(
            "split: train=%d val=%d test=%d eval_users=%d",
            len(split.train),
            len(split.val),
            len(split.test),
            len(split.eval_users),
        )
        logger.info("HIN: %s", hin)

    pretrained = None
    if pretrain:
        started = time.time()
        pretrained = pretrain_all(
            hin,
            meta_paths,
            cfg.model,
            device=device,
            seed=cfg.seed,
            logger=logger,
        )
        if logger is not None:
            logger.info("metapath2vec pre-training finished in %.1fs", time.time() - started)

    return Bundle(
        dataset=dataset,
        split=split,
        hin=hin,
        meta_paths=meta_paths,
        user_paths=user_paths,
        business_paths=business_paths,
        pretrained=pretrained,
    )


def build_hret(bundle: Bundle, cfg: Config, use_pretrained: bool = True) -> HRET:
    model = HRET(
        num_users=len(bundle.dataset.user_ids),
        num_businesses=len(bundle.dataset.business_ids),
        user_paths=bundle.user_paths,
        business_paths=bundle.business_paths,
        dim=cfg.model.dim,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        scorer=cfg.model.scorer,
        fusion=cfg.model.fusion,
        use_id_embedding=cfg.model.use_id_embedding,
        use_bias=cfg.model.use_bias,
    )
    if use_pretrained:
        if bundle.pretrained is None:
            raise ValueError("Bundle has no pre-trained embeddings")
        init = {
            (path, node_type): embedding
            for path, result in bundle.pretrained.items()
            for node_type, embedding in result.embeddings.items()
        }
        model.load_pretrained(init)
    return model


def evaluate_split(
    model: torch.nn.Module,
    bundle: Bundle,
    cfg: Config,
    which: str = "test",
    device: str | torch.device = "cpu",
) -> EvalResult:
    if which == "val":
        targets = bundle.split.val
        seen = bundle.split.seen_for_val
    elif which == "test":
        targets = bundle.split.test
        seen = bundle.split.seen_for_test
    else:
        raise ValueError(f"Unknown split: {which!r}")

    model.eval()
    device = torch.device(device)
    with torch.no_grad():
        return evaluate_ranking(
            score_fn=lambda user_ids: model.score_matrix(user_ids.to(device)),
            targets=targets,
            seen_by_user=seen,
            num_items=len(bundle.dataset.business_ids),
            ks=cfg.evaluation.ks,
            candidate_pool=cfg.evaluation.candidate_pool,
            seed=cfg.seed,
        )


def _train_and_evaluate(
    model: torch.nn.Module,
    bundle: Bundle,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
    name: str,
) -> Dict[str, object]:
    monitor = f"NDCG@{max(cfg.evaluation.ks)}"

    def val_fn(candidate: torch.nn.Module) -> EvalResult:
        return evaluate_split(candidate, bundle, cfg, which="val", device=device)

    started = time.time()
    history: TrainingHistory = train_bpr(
        model,
        bundle.split.train_by_user,
        len(bundle.dataset.business_ids),
        cfg.train,
        device=device,
        val_fn=val_fn,
        monitor=monitor,
        logger=logger,
        seed=cfg.seed,
    )
    val_result = evaluate_split(model, bundle, cfg, which="val", device=device)
    test_result = evaluate_split(model, bundle, cfg, which="test", device=device)
    logger.info("%s | val  %s", name, format_metrics(val_result.metrics))
    logger.info("%s | test %s", name, format_metrics(test_result.metrics))
    logger.info("%s | trained in %.1fs", name, time.time() - started)

    return {
        "history": history,
        "val": val_result,
        "test": test_result,
        "seconds": time.time() - started,
        "best_epoch": history.best_epoch,
    }


def _save_topk(
    model: torch.nn.Module,
    bundle: Bundle,
    cfg: Config,
    device: torch.device,
    path: Path,
    k: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad(), open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "rank", "business_id", "score"])
        users = sorted(bundle.split.test)
        for start in range(0, len(users), 256):
            batch = users[start : start + 256]
            scores = model.score_matrix(
                torch.as_tensor(batch, dtype=torch.long, device=device)
            ).cpu()
            for row, user in enumerate(batch):
                row_scores = scores[row].clone()
                seen = bundle.split.seen_for_test(user)
                if seen:
                    row_scores[torch.as_tensor(sorted(seen), dtype=torch.long)] = -torch.inf
                values, indices = torch.topk(row_scores, k=min(k, row_scores.numel()))
                for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
                    writer.writerow(
                        [
                            bundle.dataset.user_ids[user],
                            rank,
                            bundle.dataset.business_ids[index],
                            f"{value:.6f}",
                        ]
                    )


def run_hret(
    cfg: Config,
    output_dir: str | Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Train HRET end-to-end and write checkpoints, metrics and top-k lists."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or get_logger("hret", output_dir / "train.log")
    set_seed(cfg.seed)
    device = torch.device(cfg.resolve_device())
    logger.info("device: %s | seed: %d", device, cfg.seed)
    logger.info("config: %s", cfg.to_dict())

    bundle = prepare_bundle(cfg, device=device, logger=logger)
    model = build_hret(bundle, cfg).to(device)
    logger.info(
        "HRET parameters: %d | user paths=%s | business paths=%s",
        sum(p.numel() for p in model.parameters()),
        bundle.user_paths,
        bundle.business_paths,
    )

    results = _train_and_evaluate(model, bundle, cfg, device, logger, "HRET")
    history: TrainingHistory = results["history"]
    val_result: EvalResult = results["val"]
    test_result: EvalResult = results["test"]

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "user_ids": bundle.dataset.user_ids,
        "business_ids": bundle.dataset.business_ids,
        "user_paths": bundle.user_paths,
        "business_paths": bundle.business_paths,
        "val_metrics": val_result.metrics,
        "test_metrics": test_result.metrics,
    }
    torch.save(checkpoint, output_dir / "hret.pt")
    _save_topk(model, bundle, cfg, device, output_dir / "top10_test.csv")

    metrics_payload = {
        "model": "HRET",
        "dataset": bundle.dataset.summary(),
        "config": cfg.to_dict(),
        "val": val_result.metrics,
        "test": test_result.metrics,
        "best_epoch": history.best_epoch,
        "epochs": history.epochs,
        "seconds": results["seconds"],
    }
    save_json(metrics_payload, output_dir / "metrics.json")
    save_json(cfg.to_dict(), output_dir / "config.json")
    logger.info("artefacts written to %s", output_dir)
    return metrics_payload


def run_benchmark(
    cfg: Config,
    output_dir: str | Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare popularity, BPR-MF and HRET under the same split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or get_logger("hret-benchmark", output_dir / "benchmark.log")

    set_seed(cfg.seed)
    device = torch.device(cfg.resolve_device())
    bundle = prepare_bundle(cfg, device=device, logger=logger)
    results: Dict[str, Dict[str, float]] = {}

    # Popularity -------------------------------------------------------
    popularity = PopularityRecommender(
        len(bundle.dataset.business_ids),
        interaction_counts(bundle.split.train_by_user),
    ).to(device)
    val = evaluate_split(popularity, bundle, cfg, "val", device)
    test = evaluate_split(popularity, bundle, cfg, "test", device)
    results["Popularity"] = {"val": val.metrics, "test": test.metrics}
    logger.info("Popularity | val %s", format_metrics(val.metrics))
    logger.info("Popularity | test %s", format_metrics(test.metrics))

    # BPR-MF -----------------------------------------------------------
    set_seed(cfg.seed)
    bprmf = BPRMF(
        len(bundle.dataset.user_ids),
        len(bundle.dataset.business_ids),
        dim=cfg.model.dim,
        dropout=cfg.model.dropout,
    ).to(device)
    bprmf_result = _train_and_evaluate(bprmf, bundle, cfg, device, logger, "BPR-MF")
    results["BPR-MF"] = {
        "val": bprmf_result["val"].metrics,
        "test": bprmf_result["test"].metrics,
        "best_epoch": bprmf_result["best_epoch"],
    }

    # HRET -------------------------------------------------------------
    # Run HRET in its own pipeline so that the benchmark numbers are exactly
    # the ones produced by ``scripts/train.py`` (same pre-training, same RNG
    # stream, same early-stopping protocol).
    hret_metrics = run_hret(cfg, output_dir / "hret", logger=logger)
    results["HRET"] = {
        "val": hret_metrics["val"],
        "test": hret_metrics["test"],
        "best_epoch": hret_metrics["best_epoch"],
    }

    payload = {
        "dataset": bundle.dataset.summary(),
        "config": cfg.to_dict(),
        "results": results,
    }
    save_json(payload, output_dir / "benchmark.json")

    top_k = max(cfg.evaluation.ks)
    logger.info("")
    logger.info("%-12s %10s %10s %10s", "model", f"HR@{top_k}", f"NDCG@{top_k}", "MRR")
    for name in ("Popularity", "BPR-MF", "HRET"):
        metrics = results[name]["test"]
        logger.info(
            "%-12s %10.4f %10.4f %10.4f",
            name,
            metrics.get(f"HR@{top_k}", 0.0),
            metrics.get(f"NDCG@{top_k}", 0.0),
            metrics.get("MRR", 0.0),
        )
    return results
