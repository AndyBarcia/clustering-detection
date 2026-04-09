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
    img_shape: Tuple[int, int]
    sig_embs: Optional[torch.Tensor] = None        # [L, B, Nq, S]
    seed_logits: Optional[torch.Tensor] = None     # [B, Nq]
    seed_scores: Optional[torch.Tensor] = None     # [B, Nq]
    influence_preds: Optional[torch.Tensor] = None # [L, B, Nq]
