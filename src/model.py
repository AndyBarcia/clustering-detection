import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import copy
import contextlib

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

    def _ttt_block(self, tgt, sim_head):
        if sim_head is None or self.ttt_steps <= 0:
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
            
            detached_states = {k: v.detach() for k, v in sim_head.named_parameters()}

            for _ in range(self.ttt_steps):
                sim_logits = functional_call(sim_head, detached_states, (intermediate_q[-1],))
                sim_scores = torch.sigmoid(sim_logits.squeeze(-1))  
                                
                #sim_scores = torch.sigmoid(sim_head(intermediate_q[-1]).squeeze(-1))  
                inner_loss = 1.0 - sim_scores
                
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
        sim_head=None
    ):
        
        tgt = self._self_attention_block(tgt, tgt_mask, tgt_key_padding_mask)
        tgt = self._cross_attention_block(tgt, memory, memory_mask, memory_key_padding_mask)
        tgt, intermediate_ttt_q = self._ttt_block(tgt, sim_head)
        tgt = self._ff_block(tgt)

        return tgt, intermediate_ttt_q


class TTTTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, **kwargs):
        output = tgt
        ttt_output = []
        all_outputs = [] # Track outputs from all layers
        
        for mod in self.layers:
            output, intermediate_ttt_q = mod(output, memory, **kwargs)
            ttt_output.append(intermediate_ttt_q)
            all_outputs.append(output)

        intermediate_ttt_q = torch.cat(ttt_output, dim=0) # (L*TTTSteps, B, Q, C)
        all_outputs = torch.stack(all_outputs, dim=0)     # (L, B, Q, C)
        
        return all_outputs, intermediate_ttt_q


class CustomMask2Former(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        hidden_dim = cfg.backbone.hidden_dim
        num_classes = cfg.heads.num_classes
        sig_dim = cfg.heads.sig_dim
        num_queries = cfg.decoder.num_queries
        num_layers = cfg.decoder.num_layers

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.sig_dim = sig_dim
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
        )
        self.transformer_decoder = TTTTransformerDecoder(decoder_layer, num_layers=num_layers)

        self.layer_importance = nn.Parameter(torch.zeros(num_layers))

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
        self.sig_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )
        self.sim_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.margin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.gt_cls_embed = nn.Embedding(num_classes, hidden_dim)
        self.gt_query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        spatial_tokens = cfg.spatial_hw * cfg.spatial_hw
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_tokens, hidden_dim) * 0.02)

        if cfg.learned_alpha:
            self.alpha_focal = nn.Parameter(torch.tensor(cfg.alpha_focal, dtype=torch.float32))
        else:
            self.alpha_focal = cfg.alpha_focal

        self.compact_margin = cfg.compact_margin
        self.w_proto_ttt = cfg.w_proto_ttt

    def _build_memory(self, images):
        features = self.backbone(images)
        B, C, H_f, W_f = features.shape
        memory = features.view(B, C, -1).permute(0, 2, 1)
        memory = memory + self.spatial_pos_embed[:, :memory.shape[1], :]
        return features, memory

    def _decode_queries(self, memory, query_embed=None):
        B = memory.shape[0]
        if query_embed is None:
            query_embed = self.queries.weight.unsqueeze(0).repeat(B, 1, 1)

        q_dec_all, intermediate_ttt_q = self.transformer_decoder(
            tgt=query_embed,
            memory=memory,
            sim_head=self.sim_head,
        )
        return q_dec_all, intermediate_ttt_q

    def encode_gts(self, memory, features, masks, labels, pad_mask):
        B_val, M_max = masks.shape[:2]
        _, C, Hf, Wf = features.shape

        masks_small = F.interpolate(masks.float(), size=(Hf, Wf), mode='bilinear', align_corners=False)
        denom = masks_small.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled_feat = torch.einsum('bmhw,bchw->bmc', masks_small, features) / denom

        cls_emb = self.gt_cls_embed(labels)
        query_init = self.gt_query_proj(torch.cat([pooled_feat, cls_emb], dim=-1))

        attn_mask = (masks_small.flatten(2) < 0.5)
        all_masked = attn_mask.all(dim=2, keepdim=True)
        attn_mask = attn_mask.masked_fill(all_masked, False)
        attn_mask_rep = attn_mask.repeat_interleave(4, dim=0)

        q_gt_all, _ = self.transformer_decoder(
            tgt=query_init,
            memory=memory,
            memory_mask=attn_mask_rep,
            sim_head=self.sim_head,
        )
        q_gt = q_gt_all[-1]

        sig = self.sig_head(q_gt)
        return F.normalize(sig, p=2, dim=-1)

    def _run_heads(self, q):
        mask_embs = self.mask_head(q)
        cls_preds = self.cls_head(q)
        sig_embs = F.normalize(self.sig_head(q), p=2, dim=-1)
        sim_scores = torch.sigmoid(self.sim_head(q).squeeze(-1))
        margin_preds = torch.sigmoid(self.margin_head(q).squeeze(-1))
        return mask_embs, cls_preds, sig_embs, sim_scores, margin_preds

    def forward(self, images: torch.Tensor) -> RawOutputs:
        H_img, W_img = images.shape[-2:]

        features, memory = self._build_memory(images)
        q_dec_all, intermediate_ttt_q = self._decode_queries(memory)

        mask_embs, cls_preds, sig_embs, sim_scores, margin_preds = self._run_heads(q_dec_all)

        return RawOutputs(
            features=features,
            memory=memory,
            queries=q_dec_all,
            intermediate_ttt_q=intermediate_ttt_q,
            mask_embs=mask_embs,
            cls_preds=cls_preds,
            sig_embs=sig_embs,
            sim_scores=sim_scores,
            margin_preds=margin_preds,
            layer_importance=F.softmax(self.layer_importance, dim=0),
            img_shape=(H_img, W_img),
        )
