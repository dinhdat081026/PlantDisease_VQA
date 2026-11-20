"""Central configuration. Keep the seed literal exclusively in this module."""
from dataclasses import dataclass
from typing import Literal

SEED = 59
IMAGE_MODEL = "vit_base_patch16_224"
TEXT_MODEL = "bert-base-uncased"
IGNORE_INDEX = -100
DATASET_MODES = ("mini", "class", "full")
ABLATIONS = ("full", "frozen", "question_only", "image_only")

@dataclass(frozen=True)
class Config:
    dataset_mode: Literal["mini", "class", "full"] = "mini"
    ablation: Literal["full", "frozen", "question_only", "image_only"] = "full"
    validation_fraction: float = 0.2
    class_name: str | None = None

    def __post_init__(self):
        if self.dataset_mode not in DATASET_MODES:
            raise ValueError("Unsupported dataset mode")
        if self.ablation not in ABLATIONS:
            raise ValueError("Unsupported ablation")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.dataset_mode == "class" and not (self.class_name or "").strip():
            raise ValueError("class mode requires class_name")
