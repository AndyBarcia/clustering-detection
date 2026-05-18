import torch
import torch.nn as nn
from typing import Optional
from dataclasses import asdict

from .model import build_model
from .criterion import PanopticCriterion
from .predictor import build_predictor
from .dataset import DEFAULT_INSTANCE_PALETTE_RGB
from .config import (
    PrototypeInferenceConfig, 
    PanopticSystemConfig, 
    ModelConfig, 
    LossConfig, 
    dataclass_from_dict
)


class PanopticSystem(nn.Module):
    def __init__(self, cfg: PanopticSystemConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.criterion = PanopticCriterion(cfg.loss, model_variant=cfg.model.variant)
        self.predictor = build_predictor(cfg.inference, cfg.model.variant)

    @property
    def supports_gt_prototypes(self) -> bool:
        return bool(getattr(self.model, "supports_gt_prototypes", False))

    def set_inference_config(self, inference_cfg: PrototypeInferenceConfig):
        self.cfg.inference = inference_cfg
        self.predictor = build_predictor(inference_cfg, self.cfg.model.variant)

    def _resolve_ttt_steps(self, inference_cfg: Optional[PrototypeInferenceConfig]) -> Optional[int]:
        cfg = self.cfg.inference if inference_cfg is None else inference_cfg
        return cfg.ttt_steps

    def training_step(self, images, targets):
        raw = self.model(images)
        return self.criterion(self.model, raw, targets)

    @torch.no_grad()
    def predict(self, images, inference_cfg: Optional[PrototypeInferenceConfig] = None):
        raw = self.model(images, ttt_steps_override=self._resolve_ttt_steps(inference_cfg))
        predictor = self.predictor if inference_cfg is None else build_predictor(inference_cfg, self.cfg.model.variant)
        return predictor.predict_from_raw(self.model, raw)

    @torch.no_grad()
    def predict_with_gt_prototypes(self, images, targets, inference_cfg: Optional[PrototypeInferenceConfig] = None):
        raw = self.model(images, ttt_steps_override=self._resolve_ttt_steps(inference_cfg))
        predictor = self.predictor if inference_cfg is None else build_predictor(inference_cfg, self.cfg.model.variant)
        return predictor.predict_from_raw_with_gt_prototypes(self.model, raw, targets)


def save_system_checkpoint(system: PanopticSystem, path: str, optimizer=None, extra: Optional[dict] = None):
    ckpt = {
        "model_state_dict": system.model.state_dict(),
        "model_config": asdict(system.cfg.model),
        "loss_config": asdict(system.cfg.loss),
        "inference_config": asdict(system.cfg.inference),
        "dataset_palette": [list(color) for color in DEFAULT_INSTANCE_PALETTE_RGB],
        "extra": extra or {},
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_system_checkpoint(path: str, map_location="cpu", inference_override: Optional[PrototypeInferenceConfig] = None, strict: bool = True):
    ckpt = torch.load(path, map_location=map_location)
    model_state_dict = dict(ckpt["model_state_dict"])
    model_state_dict.pop("layer_importance", None)
    seed_head_keys = [key for key in model_state_dict if key.startswith("seed_head.")]
    if seed_head_keys and "seed_logits_grid" not in model_state_dict:
        for key in seed_head_keys:
            model_state_dict.pop(key, None)

    model_cfg = dataclass_from_dict(ModelConfig, ckpt["model_config"])
    loss_cfg = dataclass_from_dict(LossConfig, ckpt["loss_config"])

    if inference_override is None:
        inference_cfg = dataclass_from_dict(PrototypeInferenceConfig, ckpt["inference_config"])
    else:
        inference_cfg = inference_override

    cfg = PanopticSystemConfig(
        model=model_cfg,
        loss=loss_cfg,
        inference=inference_cfg,
    )

    system = PanopticSystem(cfg)
    migrated_fixed_seed_grid = bool(seed_head_keys) and "seed_logits_grid" not in ckpt["model_state_dict"]
    seed_grid = model_state_dict.get("seed_logits_grid")
    if seed_grid is not None and seed_grid.shape != system.model.seed_logits_grid.shape:
        if seed_grid.dim() == 1 and seed_grid.shape[0] == system.model.seed_logits_grid.shape[1]:
            model_state_dict["seed_logits_grid"] = seed_grid.unsqueeze(0).expand_as(system.model.seed_logits_grid).clone()
        else:
            model_state_dict.pop("seed_logits_grid", None)
            migrated_fixed_seed_grid = True
    missing_attn_residuals = (
        strict and
        not any(key.startswith("transformer_decoder.attn_residual_queries") for key in model_state_dict)
    )
    allow_partial_load = missing_attn_residuals or migrated_fixed_seed_grid
    incompatible = system.model.load_state_dict(model_state_dict, strict=False if allow_partial_load else strict)

    if allow_partial_load:
        missing = [
            k for k in incompatible.missing_keys
            if not k.startswith("transformer_decoder.attn_residual_") and k != "seed_logits_grid"
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint load failed. Missing keys: {missing}. Unexpected keys: {incompatible.unexpected_keys}"
            )
    return system, ckpt
