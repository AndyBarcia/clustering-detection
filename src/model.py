import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import copy
import contextlib
from typing import Optional

from .config import ModelConfig
from .backbone import TinyDetEncoder
from .outputs import RawOutputs
from .signature_ops import pairwise_distance, pairwise_similarity


class TransformerDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer that interleaves a TTT update step 
    between the cross-attention and the feedforward network (MLP).
    Follows norm_first=True architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="gelu", ttt_steps=3, ttt_lr=0.1, ttt_momentum=0.8):
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

    def _ttt_block(self, tgt, seed_head):
        if seed_head is None or self.ttt_steps <= 0:
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
            
            detached_states = {k: v.detach() for k, v in seed_head.named_parameters()}

            for _ in range(self.ttt_steps):
                seed_logits = functional_call(seed_head, detached_states, (intermediate_q[-1],))
                seed_scores = torch.sigmoid(seed_logits.squeeze(-1))
                inner_loss = 1.0 - seed_scores
                
                # Propperly scale losses based on batch size.
                inner_loss = inner_loss.view(tgt.shape[0],-1).mean(dim=-1).sum()

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
        seed_head=None
    ):
        
        tgt = self._self_attention_block(tgt, tgt_mask, tgt_key_padding_mask)
        tgt = self._cross_attention_block(tgt, memory, memory_mask, memory_key_padding_mask)
        tgt, intermediate_ttt_q = self._ttt_block(tgt, seed_head)
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

    def forward(self, tgt, memory, memory_mask_fn=None, memory_sequence=None, **kwargs):
        output = tgt
        history = [tgt]
        ttt_output = []
        memory_sequence = memory_sequence or [memory] * self.num_layers
        
        for layer_idx, mod in enumerate(self.layers):
            layer_input = self._depth_attention_residual(history, layer_idx)
            layer_kwargs = dict(kwargs)
            layer_memory = memory_sequence[layer_idx % len(memory_sequence)]
            if memory_mask_fn is not None:
                dynamic_memory_mask = memory_mask_fn(layer_input, layer_idx, layer_memory)
                if dynamic_memory_mask is not None:
                    layer_kwargs["memory_mask"] = dynamic_memory_mask
            output, intermediate_ttt_q = mod(layer_input, layer_memory, **layer_kwargs)
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

        self.backbone = TinyDetEncoder(
            in_ch=cfg.backbone.in_channels,
            backbone_channels=cfg.backbone.channels,
            fpn_channels=hidden_dim,
            norm=cfg.backbone.norm,
            act=cfg.backbone.act,
        )
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

        self.feature_levels = ("p2", "p3", "p4", "p5")
        self.decoder_feature_levels = ("p3", "p4", "p5")
        self.level_embed = nn.Parameter(torch.randn(len(self.feature_levels), hidden_dim) * 0.02)

        spatial_hw = cfg.spatial_hw
        self.spatial_pos_embeds = nn.ParameterDict({
            "p2": nn.Parameter(torch.randn(1, spatial_hw * spatial_hw, hidden_dim) * 0.02),
            "p3": nn.Parameter(torch.randn(1, (spatial_hw // 2) * (spatial_hw // 2), hidden_dim) * 0.02),
            "p4": nn.Parameter(torch.randn(1, (spatial_hw // 4) * (spatial_hw // 4), hidden_dim) * 0.02),
            "p5": nn.Parameter(torch.randn(1, (spatial_hw // 8) * (spatial_hw // 8), hidden_dim) * 0.02),
        })

    def _decoder_seed_head(self):
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

    def _add_level_position(self, feat: torch.Tensor, level_name: str) -> torch.Tensor:
        _, _, height, width = feat.shape
        pos_tokens = self.spatial_pos_embeds[level_name][:, : height * width, :]
        pos_features = pos_tokens.permute(0, 2, 1).reshape(1, self.hidden_dim, height, width)
        level_idx = self.feature_levels.index(level_name)
        return feat + pos_features + self.level_embed[level_idx].view(1, self.hidden_dim, 1, 1)

    def _build_memory(self, images):
        pyramid_feats = self.backbone(images)
        enriched_feats = {
            level_name: self._add_level_position(feat, level_name)
            for level_name, feat in pyramid_feats.items()
        }
        memory_levels = [
            enriched_feats[level_name].flatten(2).transpose(1, 2)
            for level_name in self.decoder_feature_levels
        ]
        return enriched_feats, memory_levels

    def _predict_mask_logits(self, q: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        mask_embs = self.mask_head(q)
        return torch.einsum("bqc,bchw->bqhw", mask_embs, features)

    def _decoder_memory_mask(self, query_state: torch.Tensor, features: torch.Tensor, memory_hw=None):
        return None

    def _decode_queries(self, memory_levels, feature_maps, query_embed=None, ttt_steps_override: Optional[int] = None):
        reference_features = feature_maps[self.feature_levels[0]]
        B = reference_features.shape[0]
        if query_embed is None:
            query_embed = self.queries.weight.unsqueeze(0).repeat(B, 1, 1)

        def memory_mask_fn(query_state, _layer_idx, layer_memory):
            level_name = self.decoder_feature_levels[_layer_idx % len(self.decoder_feature_levels)]
            level_features = feature_maps[level_name]
            num_tokens = layer_memory.shape[1]
            memory_hw = None
            side = int(num_tokens ** 0.5)
            if side * side == num_tokens:
                memory_hw = (side, side)
            return self._decoder_memory_mask(query_state, level_features, memory_hw)

        with self._temporary_ttt_steps(ttt_steps_override):
            q_dec_all, intermediate_ttt_q = self.transformer_decoder(
                tgt=query_embed,
                memory=memory_levels[0],
                memory_sequence=memory_levels,
                memory_mask_fn=memory_mask_fn,
                seed_head=self._decoder_seed_head(),
            )
        return q_dec_all, intermediate_ttt_q

    def _run_heads(self, q):
        mask_embs = self.mask_head(q)
        cls_preds = self.cls_head(q)
        return mask_embs, cls_preds, None, None, None, None

    def forward(self, images: torch.Tensor, ttt_steps_override: Optional[int] = None) -> RawOutputs:
        H_img, W_img = images.shape[-2:]

        enriched_feats, memory_levels = self._build_memory(images)
        features = enriched_feats["p2"]
        memory = memory_levels[0]
        q_dec_all, intermediate_ttt_q = self._decode_queries(
            memory_levels,
            enriched_feats,
            ttt_steps_override=ttt_steps_override,
        )

        mask_embs, cls_preds, sig_embs, seed_logits, seed_scores, influence_preds = self._run_heads(q_dec_all)

        return RawOutputs(
            features=features,
            feature_maps=enriched_feats,
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
        )


class CustomMask2Former(Mask2FormerBase):
    supports_gt_prototypes = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        hidden_dim = cfg.backbone.hidden_dim
        num_classes = cfg.heads.num_classes
        sig_dim = cfg.heads.sig_dim

        self.sig_dim = sig_dim
        self.signature_normalize = cfg.heads.normalize_signatures
        self.aggregation_similarity_metric = cfg.heads.aggregation_similarity_metric
        self.identity_similarity_metric = cfg.heads.identity_similarity_metric

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

        self.gt_cls_proj = nn.Embedding(num_classes, sig_dim)
        self.gt_bbox_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )
        self.gt_mask_context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )

    def _decoder_seed_head(self):
        return self.seed_head

    def _decoder_memory_mask(self, query_state: torch.Tensor, features: torch.Tensor, memory_hw=None):
        mask_logits = self._predict_mask_logits(query_state, features)
        if memory_hw is not None and mask_logits.shape[-2:] != memory_hw:
            mask_logits = F.interpolate(mask_logits, size=memory_hw, mode="bilinear", align_corners=False)
        mask_bias = mask_logits.flatten(2)
        mask_bias = mask_bias - mask_bias.mean(dim=-1, keepdim=True)
        mask_bias = mask_bias / mask_bias.std(dim=-1, keepdim=True).clamp_min(1e-6)

        num_heads = self.transformer_decoder.layers[0].multihead_attn.num_heads
        return mask_bias.repeat_interleave(num_heads, dim=0)

    def prepare_signature_embeddings(self, signatures: torch.Tensor) -> torch.Tensor:
        if not self.signature_normalize or signatures.shape[-1] == 0:
            return signatures
        return F.normalize(signatures, p=2, dim=-1)

    def aggregation_similarity(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        clamp: bool = False,
    ) -> torch.Tensor:
        return pairwise_similarity(
            lhs,
            rhs,
            metric=self.aggregation_similarity_metric,
            clamp=clamp,
        )

    def aggregation_distance(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        clamp: bool = False,
    ) -> torch.Tensor:
        return pairwise_distance(
            lhs,
            rhs,
            metric=self.aggregation_similarity_metric,
            clamp=clamp,
        )

    def identity_similarity(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        clamp: bool = False,
    ) -> torch.Tensor:
        return pairwise_similarity(
            lhs,
            rhs,
            metric=self.identity_similarity_metric,
            clamp=clamp,
        )

    def identity_distance(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        clamp: bool = False,
    ) -> torch.Tensor:
        return pairwise_distance(
            lhs,
            rhs,
            metric=self.identity_similarity_metric,
            clamp=clamp,
        )

    def _encode_gt_mask_context(self, feature_map: torch.Tensor, masks: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature_map.shape
        masks_small = F.interpolate(masks.float(), size=(height, width), mode="bilinear", align_corners=False)
        masks_small = masks_small * pad_mask[:, :, None, None].to(dtype=feature_map.dtype)

        inside_denom = masks_small.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled_inside = torch.einsum("bmhw,bchw->bmc", masks_small, feature_map) / inside_denom

        outside_masks = (1.0 - masks_small) * pad_mask[:, :, None, None].to(dtype=feature_map.dtype)
        outside_denom = outside_masks.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled_outside = torch.einsum("bmhw,bchw->bmc", outside_masks, feature_map) / outside_denom

        pooled_context = torch.cat([pooled_inside, pooled_outside], dim=-1)
        return self.gt_mask_context_proj(pooled_context)

    def encode_gts(self, memory, features, masks, labels, pad_mask, ttt_steps_override: Optional[int] = None):
        del memory, ttt_steps_override

        if isinstance(features, dict):
            feature_maps = [features[level_name] for level_name in self.feature_levels if level_name in features]
            reference_features = feature_maps[0]
        else:
            feature_maps = [features]
            reference_features = features

        _, _, Hf, Wf = reference_features.shape

        masks_small = F.interpolate(masks.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        masks_small = masks_small * pad_mask[:, :, None, None].to(dtype=reference_features.dtype)
        denom = masks_small.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)

        ys = torch.linspace(0.0, 1.0, Hf, device=reference_features.device, dtype=reference_features.dtype).view(1, 1, Hf, 1)
        xs = torch.linspace(0.0, 1.0, Wf, device=reference_features.device, dtype=reference_features.dtype).view(1, 1, 1, Wf)
        cx = (masks_small * xs).flatten(2).sum(dim=2, keepdim=True) / denom
        cy = (masks_small * ys).flatten(2).sum(dim=2, keepdim=True) / denom

        binary_masks = masks_small >= 0.5
        cols = binary_masks.any(dim=2)
        rows = binary_masks.any(dim=3)
        x_min = cols.float().argmax(dim=2)
        x_max = (Wf - 1) - cols.flip(2).float().argmax(dim=2)
        y_min = rows.float().argmax(dim=2)
        y_max = (Hf - 1) - rows.flip(2).float().argmax(dim=2)
        valid_boxes = pad_mask & binary_masks.flatten(2).any(dim=2)

        width = torch.where(valid_boxes, (x_max - x_min + 1).to(reference_features.dtype), torch.zeros_like(cx.squeeze(-1)))
        height = torch.where(valid_boxes, (y_max - y_min + 1).to(reference_features.dtype), torch.zeros_like(cy.squeeze(-1)))
        bbox_features = torch.stack(
            [
                cx.squeeze(-1),
                cy.squeeze(-1),
                width / float(Wf),
                height / float(Hf),
            ],
            dim=-1,
        )

        cls_emb = self.gt_cls_proj(labels)
        bbox_emb = self.gt_bbox_proj(bbox_features)
        gt_signature = cls_emb + bbox_emb
        for feature_map in feature_maps:
            gt_signature = gt_signature + self._encode_gt_mask_context(feature_map, masks, pad_mask)
        return self.prepare_signature_embeddings(gt_signature)

    def _run_heads(self, q):
        mask_embs = self.mask_head(q)
        cls_preds = self.cls_head(q)
        sig_embs = self.prepare_signature_embeddings(self.sig_head(q))
        seed_logits = self.seed_head(q).squeeze(-1)
        seed_scores = torch.sigmoid(seed_logits)
        influence_preds = torch.sigmoid(self.influence_head(q).squeeze(-1))
        return mask_embs, cls_preds, sig_embs, seed_logits, seed_scores, influence_preds


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
