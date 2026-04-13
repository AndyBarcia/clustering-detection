import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import copy
import contextlib
from typing import Optional

from .config import ModelConfig
from .outputs import RawOutputs


class SimpleBackbone(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.net(x)



class TransformerDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer that interleaves a TTT update step 
    between the cross-attention and the feedforward network (MLP).
    Follows norm_first=True architecture.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        ttt_steps=3,
        ttt_lr=0.1,
        ttt_momentum=0.8,
        ttt_objective: str = "distance",
    ):
        super().__init__()
        # 1. Self-Attention components
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross-Attention components
        self.norm2 = nn.LayerNorm(d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # 3. TTT components
        self.ttt_steps = ttt_steps
        self.ttt_objective = ttt_objective
        raw_ttt_lr = torch.log(torch.expm1(torch.tensor(float(ttt_lr))))
        self.ttt_lr = nn.Parameter(raw_ttt_lr)
        raw_ttt_momentum = torch.logit(torch.tensor(float(ttt_momentum)))
        self.ttt_momentum = nn.Parameter(raw_ttt_momentum)
        
        # 3. Feedforward (MLP) components
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)

    def _self_attention_block(self, tgt, tgt_mask, tgt_key_padding_mask):
        tgt_norm = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm, 
            attn_mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask
        )
        return tgt + self.dropout1(tgt2)

    def _cross_attention_block(self, tgt, memory, memory_mask, memory_key_padding_mask):
        tgt_norm = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt_norm, memory, memory, 
            attn_mask=memory_mask, 
            key_padding_mask=memory_key_padding_mask
        )
        return tgt + self.dropout2(tgt2)

    def _ttt_inner_loss(self, q, ttt_heads):
        if ttt_heads is None:
            raise ValueError("TTT heads must be provided when TTT is enabled.")

        objective = self.ttt_objective.lower()
        if objective not in {"distance", "variance", "both"}:
            raise ValueError(f"Unknown TTT objective: {self.ttt_objective}")

        distance_head = ttt_heads.get("distance_head")
        if distance_head is None:
            raise ValueError("Distance head is required for TTT distance objectives.")

        detached_states = {k: v.detach() for k, v in distance_head.named_parameters()}
        distance_stats = functional_call(distance_head, detached_states, (q,))
        distance_mean = F.softplus(distance_stats[..., 0])
        distance_variance = F.softplus(distance_stats[..., 1]) + 1e-6

        loss_terms = []
        if objective in {"distance", "both"}:
            loss_terms.append(distance_mean)
        if objective in {"variance", "both"}:
            loss_terms.append(distance_variance)

        inner_loss = sum(loss_terms)
        return inner_loss.view(q.shape[0], -1).mean(dim=-1).sum()

    def _ttt_block(self, tgt, ttt_heads):
        if ttt_heads is None or self.ttt_steps <= 0:
            intermediate_q = [tgt]
            intermediate_q = torch.stack(intermediate_q, dim=0) # (TTTSteps,B,Q,C)
            return tgt, intermediate_q

        # In eval mode, we must locally enable gradients for the adaptation loop.
        # In training mode, gradients are already active.
        grad_context = contextlib.nullcontext() if self.training else torch.enable_grad()
        
        ttt_lr = F.softplus(self.ttt_lr) + 1e-8
        ttt_momentum = torch.sigmoid(self.ttt_momentum) + 1e-8

        with grad_context:
            # During evaluation, detach to prevent backpropping into frozen earlier layers.
            # During training, keep it attached for differentiable unrolling (MAML).
            #q = tgt if self.training else tgt.detach().requires_grad_(True)
            #q = tgt.detach().requires_grad_(True)
            #intermediate_q = [tgt.detach().requires_grad_(True)]
            intermediate_q = [tgt.requires_grad_(True)]
            v = torch.zeros_like(intermediate_q[-1])

            for _ in range(self.ttt_steps):
                inner_loss = self._ttt_inner_loss(intermediate_q[-1], ttt_heads)

                grad = torch.autograd.grad(
                    inner_loss, 
                    intermediate_q[-1],
                    create_graph=self.training, 
                    retain_graph=self.training, 
                    only_inputs=True
                )[0]

                v = ttt_momentum * v - ttt_lr * grad
                q_new = intermediate_q[-1] + v
                intermediate_q.append(q_new)

        intermediate_q = torch.stack(intermediate_q,dim=0) # (TTTSteps,B,Q,C)
        return intermediate_q[-1], intermediate_q

    def _ff_block(self, tgt):
        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        return tgt + self.dropout3(tgt2)

    def forward(
        self, 
        tgt, 
        memory, 
        tgt_mask=None, 
        memory_mask=None, 
        tgt_key_padding_mask=None, 
        memory_key_padding_mask=None,
        ttt_heads=None
    ):
        
        tgt = self._self_attention_block(tgt, tgt_mask, tgt_key_padding_mask)
        tgt = self._cross_attention_block(tgt, memory, memory_mask, memory_key_padding_mask)
        tgt, intermediate_ttt_q = self._ttt_block(tgt, ttt_heads)
        tgt = self._ff_block(tgt)

        return tgt, intermediate_ttt_q


class TTTTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, use_attention_residuals: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.use_attention_residuals = use_attention_residuals

        d_model = decoder_layer.self_attn.embed_dim
        self.attn_residual_queries = nn.Parameter(torch.randn(num_layers, d_model) * (d_model ** -0.5))
        self.attn_residual_key_scale = nn.Parameter(torch.ones(d_model))
        self.attn_residual_eps = 1e-6

    def _depth_attention_residual(self, history, layer_idx):
        if not self.use_attention_residuals or len(history) == 1:
            return history[-1]

        history_stack = torch.stack(history, dim=0)  # (D, B, Q, C)
        rms = history_stack.pow(2).mean(dim=-1, keepdim=True).add(self.attn_residual_eps).rsqrt()
        keys = history_stack * rms * self.attn_residual_key_scale.view(1, 1, 1, -1)
        logits = torch.einsum("c,dbqc->dbq", self.attn_residual_queries[layer_idx], keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("dbq,dbqc->bqc", weights, history_stack)

    def forward(self, tgt, memory, **kwargs):
        output = tgt
        history = [tgt]
        ttt_output = []
        
        for layer_idx, mod in enumerate(self.layers):
            layer_input = self._depth_attention_residual(history, layer_idx)
            output, intermediate_ttt_q = mod(layer_input, memory, **kwargs)
            history.append(output)
            ttt_output.append(intermediate_ttt_q)

        intermediate_ttt_q = torch.cat(ttt_output, dim=0) # (L*TTTSteps, B, Q, C)
        all_outputs = torch.stack(history[1:], dim=0)     # (L, B, Q, C)
        
        return all_outputs, intermediate_ttt_q


class Mask2FormerBase(nn.Module):
    supports_gt_prototypes = False

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        hidden_dim = cfg.backbone.hidden_dim
        num_classes = cfg.heads.num_classes
        num_queries = cfg.decoder.num_queries
        num_layers = cfg.decoder.num_layers

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_layers = num_layers

        self.backbone = SimpleBackbone(hidden_dim=hidden_dim)
        self.queries = nn.Embedding(num_queries, hidden_dim)

        dlcfg = cfg.decoder_layer
        decoder_layer = TransformerDecoderLayer(
            d_model=dlcfg.d_model,
            nhead=dlcfg.nhead,
            dim_feedforward=dlcfg.dim_feedforward,
            dropout=dlcfg.dropout,
            activation=dlcfg.activation,
            ttt_steps=dlcfg.ttt_steps,
            ttt_lr=dlcfg.ttt_lr,
            ttt_momentum=dlcfg.ttt_momentum,
            ttt_objective=dlcfg.ttt_objective,
        )
        self.transformer_decoder = TTTTransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            use_attention_residuals=cfg.decoder.use_attention_residuals,
        )

        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        spatial_tokens = cfg.spatial_hw * cfg.spatial_hw
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_tokens, hidden_dim) * 0.02)

    def _decoder_ttt_heads(self):
        return None

    @contextlib.contextmanager
    def _temporary_ttt_steps(self, ttt_steps_override: Optional[int] = None):
        if ttt_steps_override is None:
            yield
            return

        previous_steps = [layer.ttt_steps for layer in self.transformer_decoder.layers]
        try:
            for layer in self.transformer_decoder.layers:
                layer.ttt_steps = ttt_steps_override
            yield
        finally:
            for layer, previous in zip(self.transformer_decoder.layers, previous_steps):
                layer.ttt_steps = previous

    def _build_memory(self, images):
        features = self.backbone(images)
        B, C, H_f, W_f = features.shape
        memory = features.view(B, C, -1).permute(0, 2, 1)
        memory = memory + self.spatial_pos_embed[:, :memory.shape[1], :]
        return features, memory

    def _decode_queries(self, memory, query_embed=None, ttt_steps_override: Optional[int] = None):
        B = memory.shape[0]
        if query_embed is None:
            query_embed = self.queries.weight.unsqueeze(0).repeat(B, 1, 1)

        with self._temporary_ttt_steps(ttt_steps_override):
            q_dec_all, intermediate_ttt_q = self.transformer_decoder(
                tgt=query_embed,
                memory=memory,
                ttt_heads=self._decoder_ttt_heads(),
            )
        return q_dec_all, intermediate_ttt_q

    def _run_heads(self, q):
        mask_embs = self.mask_head(q)
        cls_preds = self.cls_head(q)
        return mask_embs, cls_preds, None, None, None, None, None, None

    def forward(self, images: torch.Tensor, ttt_steps_override: Optional[int] = None) -> RawOutputs:
        H_img, W_img = images.shape[-2:]

        features, memory = self._build_memory(images)
        q_dec_all, intermediate_ttt_q = self._decode_queries(memory, ttt_steps_override=ttt_steps_override)

        (
            mask_embs,
            cls_preds,
            sig_embs,
            seed_logits,
            seed_scores,
            influence_preds,
            distance_preds,
            distance_vars,
        ) = self._run_heads(q_dec_all)

        return RawOutputs(
            features=features,
            memory=memory,
            queries=q_dec_all,
            intermediate_ttt_q=intermediate_ttt_q,
            mask_embs=mask_embs,
            cls_preds=cls_preds,
            img_shape=(H_img, W_img),
            sig_embs=sig_embs,
            seed_logits=seed_logits,
            seed_scores=seed_scores,
            influence_preds=influence_preds,
            distance_preds=distance_preds,
            distance_vars=distance_vars,
        )


class CustomMask2Former(Mask2FormerBase):
    supports_gt_prototypes = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        hidden_dim = cfg.backbone.hidden_dim
        num_classes = cfg.heads.num_classes
        sig_dim = cfg.heads.sig_dim

        self.sig_dim = sig_dim

        self.sig_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )
        self.seed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.influence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self.gt_cls_embed = nn.Embedding(num_classes, hidden_dim)
        self.gt_query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if cfg.learned_alpha:
            self.alpha_focal = nn.Parameter(torch.tensor(cfg.alpha_focal, dtype=torch.float32))
        else:
            self.alpha_focal = cfg.alpha_focal

    def _decoder_ttt_heads(self):
        return {
            "distance_head": self.distance_head,
        }

    def encode_gts(self, memory, features, masks, labels, pad_mask, ttt_steps_override: Optional[int] = None):
        B_val, M_max = masks.shape[:2]
        _, C, Hf, Wf = features.shape

        masks_small = F.interpolate(masks.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        denom = masks_small.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled_feat = torch.einsum("bmhw,bchw->bmc", masks_small, features) / denom

        cls_emb = self.gt_cls_embed(labels)
        query_init = self.gt_query_proj(torch.cat([pooled_feat, cls_emb], dim=-1))

        attn_mask = masks_small.flatten(2) < 0.5
        all_masked = attn_mask.all(dim=2, keepdim=True)
        attn_mask = attn_mask.masked_fill(all_masked, False)
        attn_mask_rep = attn_mask.repeat_interleave(4, dim=0)

        with self._temporary_ttt_steps(ttt_steps_override):
            q_gt_all, _ = self.transformer_decoder(
                tgt=query_init,
                memory=memory,
                memory_mask=attn_mask_rep,
                ttt_heads=self._decoder_ttt_heads(),
            )
        q_gt = q_gt_all[-1]

        sig = self.sig_head(q_gt)
        return F.normalize(sig, p=2, dim=-1)

    def _run_heads(self, q):
        mask_embs = self.mask_head(q)
        cls_preds = self.cls_head(q)
        sig_embs = F.normalize(self.sig_head(q), p=2, dim=-1)
        seed_logits = self.seed_head(q).squeeze(-1)
        seed_scores = torch.sigmoid(seed_logits)
        influence_preds = torch.sigmoid(self.influence_head(q).squeeze(-1))
        distance_stats = self.distance_head(q)
        distance_preds = F.softplus(distance_stats[..., 0])
        distance_vars = F.softplus(distance_stats[..., 1]) + 1e-6
        return (
            mask_embs,
            cls_preds,
            sig_embs,
            seed_logits,
            seed_scores,
            influence_preds,
            distance_preds,
            distance_vars,
        )


class StandardMask2Former(Mask2FormerBase):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.transformer_decoder.use_attention_residuals = False
        self.transformer_decoder.attn_residual_queries = None
        self.transformer_decoder.attn_residual_key_scale = None
        for layer in self.transformer_decoder.layers:
            layer.ttt_steps = 0
            layer.ttt_lr = None
            layer.ttt_momentum = None


def build_model(cfg: ModelConfig) -> Mask2FormerBase:
    variant = cfg.variant.lower()
    if variant == "standard_mask2former":
        return StandardMask2Former(cfg)
    if variant == "clustered":
        return CustomMask2Former(cfg)
    raise ValueError(f"Unknown model variant: {cfg.variant}")
