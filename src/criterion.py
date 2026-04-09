import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .config import LossConfig
from .model import CustomMask2Former, Mask2FormerBase
from .outputs import RawOutputs
from .query_graph import compute_local_maximum_margin


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


def dice_loss_from_logits(mask_logits: torch.Tensor, mask_targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(mask_logits).flatten(1)
    targets = mask_targets.flatten(1)

    intersection = 2.0 * (probs * targets).sum(dim=-1)
    denominator = probs.sum(dim=-1) + targets.sum(dim=-1)
    return 1.0 - (intersection + eps) / (denominator + eps)


@torch.no_grad()
def pairwise_mask_bce_cost(mask_logits: torch.Tensor, gt_masks: torch.Tensor):
    num_queries = mask_logits.shape[0]
    num_gt = gt_masks.shape[0]
    if num_queries == 0 or num_gt == 0:
        return mask_logits.new_zeros((num_queries, num_gt))

    expanded_logits = mask_logits[:, None].expand(-1, num_gt, -1, -1)
    expanded_targets = gt_masks[None].expand(num_queries, -1, -1, -1)
    bce = F.binary_cross_entropy_with_logits(expanded_logits, expanded_targets, reduction="none")
    return bce.flatten(2).mean(dim=-1)


@torch.no_grad()
def pairwise_mask_dice_cost(mask_logits: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1e-6):
    num_queries = mask_logits.shape[0]
    num_gt = gt_masks.shape[0]
    if num_queries == 0 or num_gt == 0:
        return mask_logits.new_zeros((num_queries, num_gt))

    probs = torch.sigmoid(mask_logits).flatten(1)
    targets = gt_masks.flatten(1)

    intersection = 2.0 * torch.einsum("qc,mc->qm", probs, targets)
    denominator = probs.sum(dim=-1, keepdim=True) + targets.sum(dim=-1).unsqueeze(0)
    return 1.0 - (intersection + eps) / (denominator + eps)


@torch.no_grad()
def hungarian_mask2former_assignment(
    cls_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_masks: torch.Tensor,
    cfg: LossConfig,
):
    num_queries = cls_logits.shape[0]
    num_gt = gt_labels.shape[0]
    device = cls_logits.device

    if num_queries == 0 or num_gt == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    cls_prob = F.softmax(cls_logits, dim=-1)
    class_cost = -cls_prob[:, gt_labels]
    mask_cost = pairwise_mask_bce_cost(mask_logits, gt_masks)
    dice_cost = pairwise_mask_dice_cost(mask_logits, gt_masks)

    total_cost = (
        cfg.matcher_cost_class * class_cost
        + cfg.matcher_cost_mask_bce * mask_cost
        + cfg.matcher_cost_mask_dice * dice_cost
    )
    row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
    return (
        torch.as_tensor(row_ind, device=device, dtype=torch.long),
        torch.as_tensor(col_ind, device=device, dtype=torch.long),
    )


class ClusterPanopticCriterion(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, model: CustomMask2Former, raw: RawOutputs, targets):
        return self.compute_loss(model, raw, targets)

    def compute_loss(self, model: CustomMask2Former, raw: RawOutputs, targets):
        features = raw.features
        memory = raw.memory
        mask_embs = raw.mask_embs
        cls_preds = raw.cls_preds
        sig_embs = raw.sig_embs
        seed_logits = raw.seed_logits
        influence_preds = raw.influence_preds
        H_img, W_img = raw.img_shape

        if sig_embs is None or seed_logits is None or influence_preds is None:
            raise ValueError("Clustered criterion requires signature, seed, and influence predictions.")

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
        q_seed_scores = raw.seed_scores[:, valid_b]
        q_influence = influence_preds[:, valid_b]

        L, _, N_q, S = q_sig.shape

        q_sig_flat = q_sig.transpose(0, 1).reshape(B_val, L * N_q, S)
        q_mask_emb_flat = q_mask_emb.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_cls_flat = q_cls.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_seed_logits_flat = q_seed_logits.transpose(0, 1).reshape(B_val, L * N_q)
        q_seed_scores_flat = q_seed_scores.transpose(0, 1).reshape(B_val, L * N_q)
        q_influence_flat = q_influence.transpose(0, 1).reshape(B_val, L * N_q)

        gt_sigs_norm = model.encode_gts(memory_val, features_val, gt_masks_pad, gt_labels_pad, gt_pad_mask)
        matched_query_mask, matched_gt_indices = hungarian_seed_assignment(q_sig_flat, gt_sigs_norm, gt_pad_mask)

        sim = torch.bmm(q_sig_flat, gt_sigs_norm.transpose(1, 2))
        weights_flat = assignment_weights_with_influence(
            similarity=sim,
            influence=q_influence_flat,
            alpha=model.alpha_focal,
            valid_mask=gt_pad_mask,
        )

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

        proto_cls_flat = proto_cls[gt_pad_mask]
        gt_labels_flat = gt_labels_pad[gt_pad_mask]
        loss_cls = F.cross_entropy(proto_cls_flat, gt_labels_flat)

        mask_logits = torch.einsum("bmc,bchw->bmhw", proto_mask_emb, features_val)
        mask_logits = F.interpolate(mask_logits, size=(H_img, W_img), mode="bilinear", align_corners=False)
        neg_inf = torch.finfo(mask_logits.dtype).min
        mask_logits_masked = mask_logits.masked_fill(~gt_pad_mask[:, :, None, None], neg_inf)
        gt_mask_target = gt_masks_pad.argmax(dim=1)

        loss_mask_ce = F.cross_entropy(mask_logits_masked, gt_mask_target)
        loss_mask_iou = soft_partition_iou_loss(mask_logits, gt_masks_pad, gt_pad_mask)

        local_max_logits, _, _ = compute_local_maximum_margin(
            q_sig=q_sig_flat,
            seed_values=q_seed_logits_flat,
            min_similarity=0.0,
        )
        local_max_scores, _, _ = compute_local_maximum_margin(
            q_sig=q_sig_flat,
            seed_values=q_seed_scores_flat,
            min_similarity=0.0,
        )
        seed_targets = matched_query_mask.float()
        loss_seed = F.binary_cross_entropy_with_logits(local_max_logits, seed_targets)

        loss_seed_sig = features.sum() * 0.0
        matched_pos = matched_query_mask.nonzero(as_tuple=False)
        if matched_pos.numel() > 0:
            matched_gt = matched_gt_indices[matched_query_mask]
            matched_q_sig = q_sig_flat[matched_query_mask]
            matched_gt_sig = gt_sigs_norm[matched_pos[:, 0], matched_gt]
            cos_sim_seed = (matched_q_sig * matched_gt_sig).sum(dim=-1)
            loss_seed_sig = (1.0 - cos_sim_seed).mean()

        total_loss_mask = self.cfg.w_mask_ce * loss_mask_ce + self.cfg.w_mask_iou * loss_mask_iou

        final_loss = (
            loss_seed_sig
            + loss_cls
            + total_loss_mask
            + self.cfg.w_seed * loss_seed
            + self.cfg.w_inter * loss_inter
        )

        components = {
            "loss_total": final_loss,
            "loss_seed_sig": loss_seed_sig,
            "loss_seed": loss_seed,
            "seed_margin_mean": local_max_scores.mean().detach(),
            "loss_cls": loss_cls,
            "loss_mask_ce": loss_mask_ce,
            "loss_mask_iou": loss_mask_iou,
            "loss_mask_total": total_loss_mask,
            "loss_inter": loss_inter,
        }

        return final_loss, components


class StandardMask2FormerCriterion(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        return self.compute_loss(model, raw, targets)

    def _compute_single_layer_loss(
        self,
        features: torch.Tensor,
        q_mask_emb: torch.Tensor,
        q_cls: torch.Tensor,
        targets,
        img_shape: tuple[int, int],
    ):
        H_img, W_img = img_shape
        mask_logits = torch.einsum("bqc,bchw->bqhw", q_mask_emb, features)
        mask_logits = F.interpolate(mask_logits, size=(H_img, W_img), mode="bilinear", align_corners=False)

        num_classes = q_cls.shape[-1]
        class_weights = torch.ones(num_classes, device=features.device, dtype=features.dtype)
        class_weights[0] = self.cfg.no_object_weight

        cls_losses = []
        matched_mask_logits = []
        matched_mask_targets = []

        for b in range(features.shape[0]):
            labels = targets[b]["labels"].to(features.device)
            masks = targets[b]["masks"].to(features.device).float()
            fg_keep = labels != 0
            gt_labels = labels[fg_keep]
            gt_masks = masks[fg_keep]

            target_classes = torch.zeros(q_cls.shape[1], dtype=torch.long, device=features.device)

            if gt_labels.numel() > 0:
                matched_query_idx, matched_gt_idx = hungarian_mask2former_assignment(
                    cls_logits=q_cls[b],
                    mask_logits=mask_logits[b],
                    gt_labels=gt_labels,
                    gt_masks=gt_masks,
                    cfg=self.cfg,
                )
                target_classes[matched_query_idx] = gt_labels[matched_gt_idx]

                if matched_query_idx.numel() > 0:
                    matched_mask_logits.append(mask_logits[b, matched_query_idx])
                    matched_mask_targets.append(gt_masks[matched_gt_idx])

            cls_losses.append(F.cross_entropy(q_cls[b], target_classes, weight=class_weights))

        if cls_losses:
            loss_cls = torch.stack(cls_losses).mean()
        else:
            loss_cls = features.sum() * 0.0

        if matched_mask_logits:
            matched_logits = torch.cat(matched_mask_logits, dim=0)
            matched_targets = torch.cat(matched_mask_targets, dim=0)
            loss_mask_bce = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
            loss_mask_dice = dice_loss_from_logits(matched_logits, matched_targets).mean()
        else:
            loss_mask_bce = features.sum() * 0.0
            loss_mask_dice = features.sum() * 0.0

        return loss_cls, loss_mask_bce, loss_mask_dice

    def compute_loss(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        del model

        features = raw.features
        layer_losses = [
            self._compute_single_layer_loss(features, q_mask_emb, q_cls, targets, raw.img_shape)
            for q_mask_emb, q_cls in zip(raw.mask_embs, raw.cls_preds)
        ]

        loss_cls = torch.stack([loss[0] for loss in layer_losses]).mean()
        loss_mask_bce = torch.stack([loss[1] for loss in layer_losses]).mean()
        loss_mask_dice = torch.stack([loss[2] for loss in layer_losses]).mean()

        total_loss_mask = self.cfg.w_mask_bce * loss_mask_bce + self.cfg.w_mask_dice * loss_mask_dice
        final_loss = loss_cls + total_loss_mask
        zero = features.sum() * 0.0

        components = {
            "loss_total": final_loss,
            "loss_seed_sig": zero,
            "loss_seed": zero,
            "loss_cls": loss_cls,
            "loss_mask_bce": loss_mask_bce,
            "loss_mask_dice": loss_mask_dice,
            "loss_mask_total": total_loss_mask,
            "loss_inter": zero,
        }
        return final_loss, components


class PanopticCriterion(nn.Module):
    def __init__(self, cfg: LossConfig, model_variant: str = "clustered"):
        super().__init__()
        self.cfg = cfg
        self.model_variant = model_variant.lower()
        self.cluster_criterion = ClusterPanopticCriterion(cfg)
        self.standard_criterion = StandardMask2FormerCriterion(cfg)

    def forward(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        if self.model_variant == "standard_mask2former":
            return self.standard_criterion(model, raw, targets)
        return self.cluster_criterion(model, raw, targets)
