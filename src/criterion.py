import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.optimize import linear_sum_assignment

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


def soft_partition_iou_loss(logits, targets, valid_mask, eps=1e-6):
    """Computes a soft IoU loss over a per-pixel instance partition."""
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~valid_mask[:, :, None, None], neg_inf)
    preds = F.softmax(masked_logits, dim=1)

    preds = preds * valid_mask[:, :, None, None]
    targets = targets * valid_mask[:, :, None, None]

    preds = preds.flatten(2)
    targets = targets.flatten(2)

    intersection = (preds * targets).sum(dim=-1)
    union = preds.sum(dim=-1) + targets.sum(dim=-1) - intersection
    iou = (intersection + eps) / (union + eps)

    if valid_mask.any():
        return 1.0 - iou[valid_mask].mean()
    return logits.sum() * 0.0


def total_variation_loss(masks: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation regularizer averaged over masks."""
    if masks.ndim != 3:
        raise ValueError(
            "Expected masks with shape (N, H, W); got "
            f"{tuple(masks.shape)} instead"
        )

    if masks.shape[0] == 0:
        return masks.sum() * 0.0

    tv_h = (masks[:, 1:, :] - masks[:, :-1, :]).abs().mean()
    tv_w = (masks[:, :, 1:] - masks[:, :, :-1]).abs().mean()
    return tv_h + tv_w


def hungarian_seed_assignment(
    q_sig_flat: torch.Tensor,
    gt_sigs_norm: torch.Tensor,
    gt_pad_mask: torch.Tensor,
):
    B, num_queries, _ = q_sig_flat.shape
    matched_query_mask = torch.zeros((B, num_queries), dtype=torch.bool, device=q_sig_flat.device)
    matched_gt_indices = torch.full((B, num_queries), -1, dtype=torch.long, device=q_sig_flat.device)

    for b in range(B):
        valid_gt_idx = torch.where(gt_pad_mask[b])[0]
        if valid_gt_idx.numel() == 0 or num_queries == 0:
            continue

        sim = torch.matmul(q_sig_flat[b], gt_sigs_norm[b, valid_gt_idx].T)
        cost = (1.0 - sim).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) == 0:
            continue

        row_ind_t = torch.as_tensor(row_ind, device=q_sig_flat.device, dtype=torch.long)
        col_ind_t = valid_gt_idx[torch.as_tensor(col_ind, device=q_sig_flat.device, dtype=torch.long)]
        matched_query_mask[b, row_ind_t] = True
        matched_gt_indices[b, row_ind_t] = col_ind_t

    return matched_query_mask, matched_gt_indices


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
        seed_logits = raw.seed_logits
        influence_preds = raw.influence_preds
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
                "loss_seed_sig": zero,
                "loss_seed": zero,
                "loss_cls": zero,
                "loss_mask_ce": zero,
                "loss_mask_iou": zero,
                "loss_mask_tv": zero,
                "loss_mask_total": zero,
                "loss_inter": zero,
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
        q_seed_logits = seed_logits[:, valid_b]
        q_influence = influence_preds[:, valid_b]

        L, _, N_q, S = q_sig.shape

        q_sig_flat = q_sig.transpose(0, 1).reshape(B_val, L * N_q, S)
        q_mask_emb_flat = q_mask_emb.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_cls_flat = q_cls.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_seed_logits_flat = q_seed_logits.transpose(0, 1).reshape(B_val, L * N_q)
        q_influence_flat = q_influence.transpose(0, 1).reshape(B_val, L * N_q)

        gt_sigs_norm = model.encode_gts(memory_val, features_val, gt_masks_pad, gt_labels_pad, gt_pad_mask)
        matched_query_mask, matched_gt_indices = hungarian_seed_assignment(q_sig_flat, gt_sigs_norm, gt_pad_mask)

        sim = torch.bmm(q_sig_flat, gt_sigs_norm.transpose(1, 2))
        sim_masked = sim.masked_fill(~gt_pad_mask.unsqueeze(1), -1.0)
        weights_raw = assignment_weights_with_influence(
            similarity=sim,
            influence=q_influence_flat,
            alpha=model.alpha_focal,
            valid_mask=gt_pad_mask,
        )
        weights_flat = weights_raw

        loss_inter = features.sum() * 0.0
        if M_max > 1:
            gt_sim = torch.bmm(gt_sigs_norm, gt_sigs_norm.transpose(1, 2))
            eye = torch.eye(M_max, dtype=torch.bool, device=features.device).unsqueeze(0)
            valid_pair_mask = gt_pad_mask.unsqueeze(2) & gt_pad_mask.unsqueeze(1)
            off_diag_mask = valid_pair_mask & ~eye

            if off_diag_mask.any():
                loss_inter = F.relu(gt_sim[off_diag_mask] - self.cfg.inter_margin).pow(2).mean()

        # Normalize (B,Q,GT) along Q dimension.
        norm_w = weights_flat / (weights_flat.sum(dim=1, keepdim=True) + 1e-6)

        proto_mask_emb = torch.bmm(norm_w.transpose(1, 2), q_mask_emb_flat)
        proto_cls = torch.bmm(norm_w.transpose(1, 2), q_cls_flat)

        proto_cls_flat = proto_cls[gt_pad_mask]
        gt_labels_flat = gt_labels_pad[gt_pad_mask]
        loss_cls = F.cross_entropy(proto_cls_flat, gt_labels_flat)

        mask_logits = torch.einsum("bmc,bchw->bmhw", proto_mask_emb, features_val)
        mask_logits = F.interpolate(mask_logits, size=(H_img, W_img), mode="bilinear", align_corners=False)
        neg_inf = torch.finfo(mask_logits.dtype).min
        mask_logits_masked = mask_logits.masked_fill(~gt_pad_mask[:, :, None, None], neg_inf)
        gt_mask_target = gt_masks_pad.argmax(dim=1)
        mask_probs = F.softmax(mask_logits_masked, dim=1)

        loss_mask_ce = F.cross_entropy(mask_logits_masked, gt_mask_target)
        loss_mask_iou = soft_partition_iou_loss(mask_logits, gt_masks_pad, gt_pad_mask)
        loss_mask_tv = total_variation_loss(mask_probs[gt_pad_mask])

        seed_targets = matched_query_mask.float()
        loss_seed = F.binary_cross_entropy_with_logits(q_seed_logits_flat, seed_targets)

        loss_seed_sig = features.sum() * 0.0
        matched_pos = matched_query_mask.nonzero(as_tuple=False)
        if matched_pos.numel() > 0:
            matched_gt = matched_gt_indices[matched_query_mask]
            matched_q_sig = q_sig_flat[matched_query_mask]
            matched_gt_sig = gt_sigs_norm[matched_pos[:, 0], matched_gt]
            cos_sim_seed = (matched_q_sig * matched_gt_sig).sum(dim=-1)
            loss_seed_sig = (1.0 - cos_sim_seed).mean()

        total_loss_mask = (
            self.cfg.w_mask_ce * loss_mask_ce
            + self.cfg.w_mask_iou * loss_mask_iou
            + self.cfg.w_mask_tv * loss_mask_tv
        )

        final_loss = (
            loss_seed_sig +
            loss_cls +
            total_loss_mask +
            self.cfg.w_seed * loss_seed +
            self.cfg.w_inter * loss_inter
        )

        components = {
            "loss_total": final_loss,
            "loss_seed_sig": loss_seed_sig,
            "loss_seed": loss_seed,
            "loss_cls": loss_cls,
            "loss_mask_ce": loss_mask_ce,
            "loss_mask_iou": loss_mask_iou,
            "loss_mask_tv": loss_mask_tv,
            "loss_mask_total": total_loss_mask,
            "loss_inter": loss_inter,
        }

        return final_loss, components
