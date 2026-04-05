from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Tuple, Union, get_origin, get_args
import torch


@dataclass
class RawOutputs:
    features: torch.Tensor          # [B, C, Hf, Wf]
    memory: torch.Tensor            # [B, HW, C]
    queries: torch.Tensor           # [L, B, Nq, C]
    intermediate_ttt_q: torch.Tensor
    mask_embs: torch.Tensor         # [L, B, Nq, C]
    cls_preds: torch.Tensor         # [L, B, Nq, K]
    sig_embs: torch.Tensor          # [L, B, Nq, S]
    seed_logits: torch.Tensor       # [L, B, Nq]
    seed_scores: torch.Tensor       # [L, B, Nq]
    influence_preds: torch.Tensor   # [L, B, Nq]
    img_shape: Tuple[int, int]
