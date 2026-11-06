"""Typed configuration objects for the HRET pipeline."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_META_PATHS = ["UBU", "USU", "URU", "BUB", "BSB"]


@dataclass
class DataConfig:
    """Where the dataset lives and how interactions are split."""

    dir: str = "data/sample"
    min_interactions: int = 3
    split_strategy: str = "random"  # random | temporal


@dataclass
class ModelConfig:
    """Hyper-parameters of the HIN encoder and the HRET ranker."""

    dim: int = 64
    meta_paths: List[str] = field(default_factory=lambda: list(DEFAULT_META_PATHS))
    walks_per_node: int = 8
    walk_length: int = 30
    window_size: int = 3
    pretrain_epochs: int = 5
    pretrain_batch_size: int = 512
    pretrain_lr: float = 0.01
    neg_samples: int = 5
    dropout: float = 0.2
    hidden_dim: int = 128
    scorer: str = "mlp"
    fusion: str = "attention"
    use_id_embedding: bool = False
    use_bias: bool = False


@dataclass
class TrainConfig:
    """Optimisation schedule for the BPR ranking objective."""

    epochs: int = 40
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    l2_reg: float = 1e-5
    grad_clip: float = 5.0
    patience: int = 10
    eval_every: int = 1


@dataclass
class EvalConfig:
    """Evaluation protocol."""

    ks: List[int] = field(default_factory=lambda: [5, 10])
    candidate_pool: int = 0


@dataclass
class Config:
    """Top-level experiment configuration."""

    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    # (De)serialisation
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        raw = raw or {}
        return cls(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self) -> "Config":
        return Config.from_dict(copy.deepcopy(self.to_dict()))

    def apply_overrides(self, overrides: List[str]) -> "Config":
        """Apply ``section.key=value`` overrides and return a new config."""
        cfg = self.copy()
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must look like section.key=value, got {item!r}")
            dotted, raw_value = item.split("=", 1)
            value = yaml.safe_load(raw_value)
            target: Any = cfg
            parts = dotted.split(".")
            for part in parts[:-1]:
                if not hasattr(target, part):
                    raise KeyError(f"Unknown configuration key: {dotted}")
                target = getattr(target, part)
            if not hasattr(target, parts[-1]):
                raise KeyError(f"Unknown configuration key: {dotted}")
            setattr(target, parts[-1], value)
        return cfg

    def resolve_device(self) -> str:
        """Resolve ``device: auto`` to the best available torch device."""
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
