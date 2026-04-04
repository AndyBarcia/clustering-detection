import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .config import LossConfig
from .model import CustomMask2Former
from .outputs import RawOutputs


def assignment_weights_with_influence(
    similarity: torch.Tensor,
    influence: torch.Tensor,
    alpha,
    valid_mask: torch.Tensor | None = None,
):
    affinity = (similarity + influence.unsqueeze(-1)).clamp(0.0, 1.0)
    if valid_mask is not None:
        affinity = affinity.masked_fill(~valid_mask.unsqueeze(1), 0.0)
    return affinity.pow(alpha)


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Optional: Focal loss is often better than BCE for extreme foreground/background imbalance"""
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def soft_iou_loss(inputs, targets, eps=1e-6):
    """Computes Soft Intersection over Union (IoU) Loss."""
    # inputs are logits, so apply sigmoid
    preds = torch.sigmoid(inputs)
    
    # Flatten spatial dimensions
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6):
    denom = weights.sum()
    if float(denom.detach().item()) <= eps:
        return values.sum() * 0.0
    return (values * weights).sum() / denom.clamp_min(eps)


class PanopticCriterion(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, model: CustomMask2Former, raw: RawOutputs, targets):
        return self.compute_loss(model, raw, targets)

    def compute_loss(self, model: CustomMask2Former, raw: RawOutputs, targets):
        features = raw.features
        memory = raw.memory
        queries = raw.queries
        mask_embs = raw.mask_embs
        cls_preds = raw.cls_preds
        sig_embs = raw.sig_embs
        sim_scores = raw.sim_scores
        influence_preds = raw.influence_preds
        margin_preds = raw.margin_preds
        seed_probs = raw.seed_probs
        graph_embs = raw.graph_embs
        intermediate_ttt_q = raw.intermediate_ttt_q
        H_img, W_img = raw.img_shape

        B = features.shape[0]

        valid_b, masks_list, labels_list = [], [], []
        for b in range(B):
            l = targets[b]["labels"].to(features.device)
            if len(l) > 0:
                valid_b.append(b)
                masks_list.append(targets[b]["masks"].to(features.device).float())
                labels_list.append(l)

        if not valid_b:
            zero = features.sum() * 0.0
            return zero, {
                "loss_total": zero,
                "loss_proto_sig": zero,
                "loss_proto_ttt": zero,
                "loss_cls": zero,
                "loss_mask_bce": zero,
                "loss_mask_iou": zero,
                "loss_mask_total": zero,
                "loss_sim": zero,
                "loss_margin": zero,
                "loss_inter": zero,
                "loss_seed_cover": zero,
                "loss_seed_sparse": zero,
                "loss_graph_pos": zero,
                "loss_graph_neg": zero,
            }

        B_val = len(valid_b)
        M_max = max(len(l) for l in labels_list)

        gt_masks_pad = torch.zeros(B_val, M_max, H_img, W_img, device=features.device)
        gt_labels_pad = torch.zeros(B_val, M_max, dtype=torch.long, device=features.device)
        gt_pad_mask = torch.zeros(B_val, M_max, dtype=torch.bool, device=features.device)

        for i, (m, l) in enumerate(zip(masks_list, labels_list)):
            M = len(l)
            gt_masks_pad[i, :M] = m
            gt_labels_pad[i, :M] = l
            gt_pad_mask[i, :M] = True

        features_val = features[valid_b]
        memory_val = memory[valid_b]

        q_sig = sig_embs[:, valid_b]
        q_mask_emb = mask_embs[:, valid_b]
        q_cls = cls_preds[:, valid_b]
        q_sim_score = sim_scores[:, valid_b]
        q_influence = influence_preds[:, valid_b]
        q_margin = margin_preds[:, valid_b]
        q_seed = seed_probs[:, valid_b]
        q_graph = graph_embs[:, valid_b]

        L, _, N_q, S = q_sig.shape
        G = q_graph.shape[-1]

        q_sig_flat = q_sig.transpose(0, 1).reshape(B_val, L * N_q, S)
        q_mask_emb_flat = q_mask_emb.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_cls_flat = q_cls.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_sim_score_flat = q_sim_score.transpose(0, 1).reshape(B_val, L * N_q)
        q_influence_flat = q_influence.transpose(0, 1).reshape(B_val, L * N_q)
        q_margin_flat = q_margin.transpose(0, 1).reshape(B_val, L * N_q)
        q_seed_flat = q_seed.transpose(0, 1).reshape(B_val, L * N_q)
        q_graph_flat = q_graph.transpose(0, 1).reshape(B_val, L * N_q, G)

        gt_sigs_norm = model.encode_gts(memory_val, features_val, gt_masks_pad, gt_labels_pad, gt_pad_mask)

        sim = torch.bmm(q_sig_flat, gt_sigs_norm.transpose(1, 2))
        sim_masked = sim.masked_fill(~gt_pad_mask.unsqueeze(1), -1.0)
        weights_raw = assignment_weights_with_influence(
            similarity=sim,
            influence=q_influence_flat,
            alpha=model.alpha_focal,
            valid_mask=gt_pad_mask,
        )
        weights_flat = weights_raw
        sim_pos = sim.clamp_min(0.0).masked_fill(~gt_pad_mask.unsqueeze(1), 0.0)
        membership = sim_pos / (sim_pos.sum(dim=2, keepdim=True) + 1e-6)
        membership_detached = membership.detach()

        top2 = sim_masked.topk(k=min(2, M_max), dim=2).values
        s1 = top2[:, :, 0]
        if M_max > 1:
            s2 = top2[:, :, 1]
            M_valid = gt_pad_mask.sum(dim=1)
            s2 = s2 * (M_valid > 1).unsqueeze(1).float()
        else:
            s2 = torch.zeros_like(s1)

        true_margin = s1 - s2
        loss_margin = F.mse_loss(q_margin_flat, true_margin.detach())

        loss_inter = features.sum() * 0.0
        if M_max > 1:
            gt_sim = torch.bmm(gt_sigs_norm, gt_sigs_norm.transpose(1, 2))
            eye = torch.eye(M_max, dtype=torch.bool, device=features.device).unsqueeze(0)
            valid_pair_mask = gt_pad_mask.unsqueeze(2) & gt_pad_mask.unsqueeze(1)
            off_diag_mask = valid_pair_mask & ~eye

            if off_diag_mask.any():
                loss_inter = F.relu(gt_sim[off_diag_mask] - self.cfg.inter_margin).pow(2).mean()

        norm_w = weights_flat / (weights_flat.sum(dim=1, keepdim=True) + 1e-6)

        proto_mask_emb = torch.bmm(norm_w.transpose(1, 2), q_mask_emb_flat)
        proto_cls = torch.bmm(norm_w.transpose(1, 2), q_cls_flat)

        proto_signature = torch.bmm(norm_w.transpose(1, 2), q_sig_flat)
        proto_signature = F.normalize(proto_signature, p=2, dim=-1)

        cos_sim_proto = (proto_signature * gt_sigs_norm).sum(dim=-1)
        loss_proto_sig = F.relu(model.compact_margin - cos_sim_proto)[gt_pad_mask].pow(2).mean()

        proto_cls_flat = proto_cls[gt_pad_mask]
        gt_labels_flat = gt_labels_pad[gt_pad_mask]
        loss_cls = F.cross_entropy(proto_cls_flat, gt_labels_flat)

        mask_logits = torch.einsum("bmc,bchw->bmhw", proto_mask_emb, features_val)
        mask_logits = F.interpolate(mask_logits, size=(H_img, W_img), mode="bilinear", align_corners=False)

        mask_logits_flat = mask_logits[gt_pad_mask]
        gt_masks_flat = gt_masks_pad[gt_pad_mask]

        loss_mask_bce = F.binary_cross_entropy_with_logits(mask_logits_flat, gt_masks_flat)
        loss_mask_iou = soft_iou_loss(mask_logits_flat, gt_masks_flat)

        true_sim_max, _ = sim_masked.max(dim=2)
        loss_sim = F.mse_loss(q_sim_score_flat, true_sim_max.detach())

        with torch.no_grad():
            sig_embs_ttt = F.normalize(model.sig_head(intermediate_ttt_q), p=2, dim=-1)
            sig_embs_ttt_val = sig_embs_ttt[:, valid_b]

        sim_scores_ttt = torch.sigmoid(model.sim_head(intermediate_ttt_q).squeeze(-1))
        sim_scores_ttt_val = sim_scores_ttt[:, valid_b]

        with torch.no_grad():
            sim_ttt = torch.einsum('tbns,bms->tbnm', sig_embs_ttt_val, gt_sigs_norm)
            sim_ttt_masked = sim_ttt.masked_fill(~gt_pad_mask.view(1, B_val, 1, M_max), -1.0)
            true_sim_ttt = sim_ttt_masked.max(dim=-1).values

        loss_sim = loss_sim + F.mse_loss(sim_scores_ttt_val, true_sim_ttt.detach())

        loss_proto_ttt = F.relu(model.compact_margin - true_sim_max).pow(model.alpha_focal).mean()
        seed_support = (membership_detached * q_seed_flat.unsqueeze(-1)).clamp(0.0, 1.0 - 1e-6)
        log_miss = torch.log1p(-seed_support).sum(dim=1)
        cover_prob = 1.0 - torch.exp(log_miss)
        loss_seed_cover = -(cover_prob.clamp_min(1e-6).log())[gt_pad_mask].mean()
        loss_seed_sparse = q_seed_flat.mean()

        query_support = membership_detached.sum(dim=2).clamp(0.0, 1.0)
        graph_cos = torch.bmm(q_graph_flat, q_graph_flat.transpose(1, 2))
        seed_pair = q_seed_flat.unsqueeze(2) * q_seed_flat.unsqueeze(1)
        support_pair = query_support.unsqueeze(2) * query_support.unsqueeze(1)
        same_prob = torch.bmm(membership_detached, membership_detached.transpose(1, 2))
        diff_prob = (support_pair - same_prob).clamp_min(0.0)

        pair_mask = ~torch.eye(L * N_q, dtype=torch.bool, device=features.device).unsqueeze(0)
        pair_mask_f = pair_mask.float()

        pos_weights = same_prob * seed_pair * pair_mask_f
        neg_weights = diff_prob * seed_pair * pair_mask_f
        pos_violation = F.relu(self.cfg.graph_pos_margin - graph_cos).pow(2)
        neg_violation = F.relu(graph_cos - self.cfg.graph_neg_margin).pow(2)
        loss_graph_pos = weighted_mean(pos_violation, pos_weights)
        loss_graph_neg = weighted_mean(neg_violation, neg_weights)

        total_loss_mask = self.cfg.w_mask_bce * loss_mask_bce + self.cfg.w_mask_iou * loss_mask_iou

        final_loss = (
            loss_proto_sig +
            self.cfg.w_proto_ttt * loss_proto_ttt +
            loss_cls +
            total_loss_mask +
            self.cfg.w_sim * loss_sim +
            self.cfg.w_margin * loss_margin +
            self.cfg.w_inter * loss_inter +
            self.cfg.w_seed_cover * loss_seed_cover +
            self.cfg.w_seed_sparse * loss_seed_sparse +
            self.cfg.w_graph_pos * loss_graph_pos +
            self.cfg.w_graph_neg * loss_graph_neg
        )

        components = {
            "loss_total": final_loss,
            "loss_proto_sig": loss_proto_sig,
            "loss_proto_ttt": loss_proto_ttt,
            "loss_cls": loss_cls,
            "loss_mask_bce": loss_mask_bce,
            "loss_mask_iou": loss_mask_iou,
            "loss_mask_total": total_loss_mask,
            "loss_sim": loss_sim,
            "loss_margin": loss_margin,
            "loss_inter": loss_inter,
            "loss_seed_cover": loss_seed_cover,
            "loss_seed_sparse": loss_seed_sparse,
            "loss_graph_pos": loss_graph_pos,
            "loss_graph_neg": loss_graph_neg,
        }

        return final_loss, components
