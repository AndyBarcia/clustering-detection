import torch
import torch.nn as nn
from typing import Optional
from dataclasses import asdict

from .model import CustomMask2Former
from .criterion import PanopticCriterion
from .predictor import ModularPrototypePredictor
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
        self.model = CustomMask2Former(cfg.model)
        self.criterion = PanopticCriterion(cfg.loss)
        self.predictor = ModularPrototypePredictor(cfg.inference)

    def set_inference_config(self, inference_cfg: PrototypeInferenceConfig):
        self.cfg.inference = inference_cfg
        self.predictor = ModularPrototypePredictor(inference_cfg)

    def training_step(self, images, targets):
        raw = self.model.forward_raw(images)
        loss = self.criterion(self.model, raw, targets)
        return loss

    @torch.no_grad()
    def predict(self, images, inference_cfg: Optional[PrototypeInferenceConfig] = None):
        raw = self.model.forward_raw(images)
        predictor = self.predictor if inference_cfg is None else ModularPrototypePredictor(inference_cfg)
        return predictor.predict_from_raw(self.model, raw)


def save_system_checkpoint(system: PanopticSystem, path: str, optimizer=None, extra: Optional[dict] = None):
    ckpt = {
        "model_state_dict": system.model.state_dict(),
        "model_config": asdict(system.cfg.model),
        "loss_config": asdict(system.cfg.loss),
        "inference_config": asdict(system.cfg.inference),
        "extra": extra or {},
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_system_checkpoint(path: str, map_location="cpu", inference_override: Optional[PrototypeInferenceConfig] = None, strict: bool = True):
    ckpt = torch.load(path, map_location=map_location)

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
    system.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return system, ckpt