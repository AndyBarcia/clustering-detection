import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .config import LossConfig
from .mask_aggregation import (
    assignment_weights_with_influence,
    aggregate_with_weights,
    normalize_assignment_weights,
    project_mask_embeddings,
)
from .model import CustomMask2Former, Mask2FormerBase
from .outputs import RawOutputs
from .signature_ops import pairwise_similarity

def soft_partition_iou_loss(logits, targets, valid_mask, eps=1e-6):
    """Computes a soft IoU loss over a per-pixel instance partition."""
    # logits/targets: [B,GT,H,W], valid_mask: [B,GT]
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
    q_sig_flat: torch.Tensor, # [B,Q,S]
    gt_sigs_norm: torch.Tensor, # [B,GT,S]
    gt_pad_mask: torch.Tensor, # [B,GT]
    q_seed_logits_flat: torch.Tensor | None = None, # [B,Q]
    *,
    similarity_metric: str = "dot",
    seed_cost_weight: float = 1.0,
):
    B, num_queries, _ = q_sig_flat.shape
    matched_query_mask = torch.zeros((B, num_queries), dtype=torch.bool, device=q_sig_flat.device)
    matched_gt_indices = torch.full((B, num_queries), -1, dtype=torch.long, device=q_sig_flat.device)

    for b in range(B):
        valid_gt_idx = torch.where(gt_pad_mask[b])[0]
        if valid_gt_idx.numel() == 0 or num_queries == 0:
            continue

        sim = pairwise_similarity(
            q_sig_flat[b],
            gt_sigs_norm[b, valid_gt_idx],
            metric=similarity_metric,
        )
        cost = 1.0 - sim
        if q_seed_logits_flat is not None and seed_cost_weight != 0.0:
            seed_cost = 1.0 - torch.sigmoid(q_seed_logits_flat[b])
            cost = cost + seed_cost_weight * seed_cost.unsqueeze(-1)
        cost = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) == 0:
            continue

        row_ind_t = torch.as_tensor(row_ind, device=q_sig_flat.device, dtype=torch.long)
        col_ind_t = valid_gt_idx[torch.as_tensor(col_ind, device=q_sig_flat.device, dtype=torch.long)]
        matched_query_mask[b, row_ind_t] = True
        matched_gt_indices[b, row_ind_t] = col_ind_t

    return matched_query_mask, matched_gt_indices


def dice_loss_from_logits(mask_logits: torch.Tensor, mask_targets: torch.Tensor, eps: float = 1e-6):
    # mask_logits/mask_targets: [N,H,W]
    probs = torch.sigmoid(mask_logits).flatten(1)
    targets = mask_targets.flatten(1)

    intersection = 2.0 * (probs * targets).sum(dim=-1)
    denominator = probs.sum(dim=-1) + targets.sum(dim=-1)
    return 1.0 - (intersection + eps) / (denominator + eps)


@torch.no_grad()
def pairwise_mask_bce_cost(mask_logits: torch.Tensor, gt_masks: torch.Tensor):
    # mask_logits: [Q,H,W], gt_masks: [GT,H,W]
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
    # mask_logits: [Q,H,W], gt_masks: [GT,H,W]
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
    cls_logits: torch.Tensor, # [Q,C]
    mask_logits: torch.Tensor, # [Q,H,W]
    gt_labels: torch.Tensor, # [GT]
    gt_masks: torch.Tensor, # [GT,H,W]
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

    def _ordered_feature_maps(
        self,
        model: Mask2FormerBase,
        raw: RawOutputs,
    ) -> list[tuple[str, torch.Tensor]]:
        return [
            (level_name, raw.feature_maps[level_name])
            for level_name in model.get_mask_loss_feature_levels()
            if level_name in raw.feature_maps
        ]

    def _compute_loss_inter(
        self,
        gt_sigs_norm: torch.Tensor, # [B,GT,S]
        gt_pad_mask: torch.Tensor, # [B,GT]
        features: torch.Tensor, # [B,C,Hf,Wf]
        identity_similarity_metric: str,
    ):
        num_gt = gt_pad_mask.shape[1]
        loss_inter = features.sum() * 0.0

        if num_gt <= 1:
            return loss_inter

        gt_sim = pairwise_similarity(
            gt_sigs_norm,
            gt_sigs_norm,
            metric=identity_similarity_metric,
        )
        eye = torch.eye(num_gt, dtype=torch.bool, device=features.device).unsqueeze(0)
        valid_pair_mask = gt_pad_mask.unsqueeze(2) & gt_pad_mask.unsqueeze(1)
        off_diag_mask = valid_pair_mask & ~eye

        if off_diag_mask.any():
            loss_inter = F.relu(gt_sim[off_diag_mask] - self.cfg.inter_margin).pow(2).mean()

        return loss_inter

    def _compute_assignment_aggregation(
        self,
        q_sig_flat: torch.Tensor, # [B,Q,S]
        q_influence_flat: torch.Tensor, # [B,Q]
        gt_sigs_norm: torch.Tensor, # [B,GT,S]
        gt_pad_mask: torch.Tensor, # [B,GT]
        q_mask_emb_flat: torch.Tensor, # [B,Q,Cm]
        q_cls_flat: torch.Tensor, # [B,Q,K]
        aggregation_similarity_metric: str,
    ):
        sim = pairwise_similarity(
            q_sig_flat,
            gt_sigs_norm,
            metric=aggregation_similarity_metric,
        )
        weights_flat = assignment_weights_with_influence(
            similarity=sim,
            influence=q_influence_flat,
            alpha=1.0,
            valid_mask=gt_pad_mask,
        )
        # norm_w: [B,Q,GT] normalized across queries so each GT builds a soft prototype.
        norm_w = normalize_assignment_weights(weights_flat, normalize_over_queries=True)

        # proto_mask_emb: [B,GT,Cm], proto_cls: [B,GT,K]
        proto_mask_emb = aggregate_with_weights(norm_w, q_mask_emb_flat)
        proto_cls = aggregate_with_weights(norm_w, q_cls_flat)

        return proto_mask_emb, proto_cls

    def _compute_aggregated_cls_loss(
        self,
        proto_cls: torch.Tensor, # [B,GT,K]
        gt_labels_pad: torch.Tensor, # [B,GT]
        gt_pad_mask: torch.Tensor, # [B,GT]
    ):
        # Keep only valid padded GT slots before classification loss.
        proto_cls_flat = proto_cls[gt_pad_mask]
        gt_labels_flat = gt_labels_pad[gt_pad_mask]
        return F.cross_entropy(proto_cls_flat, gt_labels_flat)

    def _compute_aggregated_mask_losses(
        self,
        proto_mask_emb: torch.Tensor, # [B,GT,C]
        feature_maps_val: list[tuple[str, torch.Tensor]],
        gt_masks_pad: torch.Tensor, # [B,GT,H,W]
        gt_pad_mask: torch.Tensor, # [B,GT]
        img_shape: tuple[int, int],
    ):
        level_loss_mask_ce = {}
        level_loss_mask_iou = {}

        for level_name, features_val in feature_maps_val:
            mask_logits = project_mask_embeddings(
                proto_mask_emb,
                features_val,
                img_shape,
            )

            neg_inf = torch.finfo(mask_logits.dtype).min
            mask_logits_masked = mask_logits.masked_fill(~gt_pad_mask[:, :, None, None], neg_inf)
            gt_mask_target = gt_masks_pad.argmax(dim=1)

            level_loss_mask_ce[level_name] = F.cross_entropy(mask_logits_masked, gt_mask_target)
            level_loss_mask_iou[level_name] = soft_partition_iou_loss(mask_logits, gt_masks_pad, gt_pad_mask)

        loss_mask_ce = torch.stack(list(level_loss_mask_ce.values())).mean()
        loss_mask_iou = torch.stack(list(level_loss_mask_iou.values())).mean()
        return loss_mask_ce, loss_mask_iou, level_loss_mask_ce, level_loss_mask_iou

    def _compute_mask_cls_losses(
        self,
        q_sig_flat: torch.Tensor, # [B,Q,S]
        q_influence_flat: torch.Tensor, # [B,Q]
        gt_sigs_norm: torch.Tensor, # [B,GT,S]
        gt_pad_mask: torch.Tensor, # [B,GT]
        q_mask_emb_flat: torch.Tensor, # [B,Q,Cm]
        q_cls_flat: torch.Tensor, # [B,Q,K]
        gt_labels_pad: torch.Tensor, # [B,GT]
        gt_masks_pad: torch.Tensor, # [B,GT,H,W]
        feature_maps_val: list[tuple[str, torch.Tensor]],
        img_shape: tuple[int, int],
        aggregation_similarity_metric: str,
    ):
        proto_mask_emb, proto_cls = self._compute_assignment_aggregation(
            q_sig_flat=q_sig_flat,
            q_influence_flat=q_influence_flat,
            gt_sigs_norm=gt_sigs_norm,
            gt_pad_mask=gt_pad_mask,
            q_mask_emb_flat=q_mask_emb_flat,
            q_cls_flat=q_cls_flat,
            aggregation_similarity_metric=aggregation_similarity_metric,
        )
        loss_cls = self._compute_aggregated_cls_loss(
            proto_cls, 
            gt_labels_pad, 
            gt_pad_mask
        )
        loss_mask_ce, loss_mask_iou, level_loss_mask_ce, level_loss_mask_iou = self._compute_aggregated_mask_losses(
            proto_mask_emb=proto_mask_emb,
            feature_maps_val=feature_maps_val,
            gt_masks_pad=gt_masks_pad,
            gt_pad_mask=gt_pad_mask,
            img_shape=img_shape,
        )
        return loss_cls, loss_mask_ce, loss_mask_iou, level_loss_mask_ce, level_loss_mask_iou

    def _compute_seed_losses(
        self,
        q_sig_flat: torch.Tensor, # [B,Q,S]
        gt_sigs_norm: torch.Tensor, # [B,GT,S]
        gt_pad_mask: torch.Tensor, # [B,GT]
        q_seed_logits_flat: torch.Tensor, # [B,Q]
        features: torch.Tensor, # [B,C,Hf,Wf]
        identity_similarity_metric: str,
    ):
        # q_seed_logits_flat: [B,Q], matched_query_mask: [B,Q]
        matched_query_mask, matched_gt_indices = hungarian_seed_assignment(
            q_sig_flat,
            gt_sigs_norm,
            gt_pad_mask,
            q_seed_logits_flat=q_seed_logits_flat,
            similarity_metric=identity_similarity_metric,
            seed_cost_weight=self.cfg.matcher_cost_seed,
        )

        seed_targets = matched_query_mask.float()
        loss_seed = F.binary_cross_entropy_with_logits(q_seed_logits_flat, seed_targets)

        loss_seed_sig = features.sum() * 0.0
        matched_pos = matched_query_mask.nonzero(as_tuple=False)
        if matched_pos.numel() > 0:
            matched_gt = matched_gt_indices[matched_query_mask]
            matched_q_sig = q_sig_flat[matched_query_mask]
            matched_gt_sig = gt_sigs_norm[matched_pos[:, 0], matched_gt].detach()
            matched_similarity = pairwise_similarity(
                matched_q_sig.unsqueeze(1),
                matched_gt_sig.unsqueeze(1),
                metric=identity_similarity_metric,
            ).squeeze(-1).squeeze(-1)
            loss_seed_sig = (1.0 - matched_similarity).mean()

        return loss_seed, loss_seed_sig

    def compute_loss(self, model: CustomMask2Former, raw: RawOutputs, targets):
        features = raw.features
        memory = raw.memory
        feature_maps = self._ordered_feature_maps(model, raw)
        mask_embs = raw.mask_embs
        cls_preds = raw.cls_preds
        sig_embs = raw.sig_embs
        seed_logits = raw.seed_logits
        influence_preds = raw.influence_preds
        H_img, W_img = raw.img_shape

        if sig_embs is None or seed_logits is None or influence_preds is None:
            raise ValueError("Clustered criterion requires signature, seed, and influence predictions.")

        # features: [B,C,Hf,Wf], memory: [B,N_mem,C]
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

        # Padded GT tensors: gt_masks_pad [Bv,GT,H,W], gt_labels_pad [Bv,GT], gt_pad_mask [Bv,GT]
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
        feature_maps_val = [(level_name, level_features[valid_b]) for level_name, level_features in feature_maps]

        q_sig = sig_embs[:, valid_b]
        q_mask_emb = mask_embs[:, valid_b]
        q_cls = cls_preds[:, valid_b]
        q_seed_logits = seed_logits[:, valid_b]
        q_influence = influence_preds[:, valid_b]

        # Decoder outputs are [L,Bv,Q,...]; we flatten layers and queries into a single query axis.
        L, _, N_q, S = q_sig.shape

        # q_sig_flat: [Bv,L*Q,S], q_mask_emb_flat: [Bv,L*Q,C], q_cls_flat: [Bv,L*Q,K]
        q_sig_flat = q_sig.transpose(0, 1).reshape(B_val, L * N_q, S)
        q_mask_emb_flat = q_mask_emb.transpose(0, 1).reshape(B_val, L * N_q, -1)
        q_cls_flat = q_cls.transpose(0, 1).reshape(B_val, L * N_q, -1)
        # q_seed_logits_flat/q_influence_flat: [Bv,L*Q]
        q_seed_logits_flat = q_seed_logits.transpose(0, 1).reshape(B_val, L * N_q)
        q_influence_flat = q_influence.transpose(0, 1).reshape(B_val, L * N_q)

        # gt_sigs_norm: [Bv,GT,S]
        feature_maps_dict = {
            level_name: level_features
            for level_name, level_features in feature_maps_val
        }
        gt_sigs_norm = model.encode_gts(memory_val, feature_maps_dict, gt_masks_pad, gt_labels_pad, gt_pad_mask)
        
        loss_inter = self._compute_loss_inter(
            gt_sigs_norm, 
            gt_pad_mask, 
            features,
            identity_similarity_metric=model.identity_similarity_metric,
        )
        
        loss_cls, loss_mask_ce, loss_mask_iou, level_loss_mask_ce, level_loss_mask_iou = self._compute_mask_cls_losses(
            q_sig_flat=q_sig_flat,
            q_influence_flat=q_influence_flat,
            gt_sigs_norm=gt_sigs_norm,
            gt_pad_mask=gt_pad_mask,
            q_mask_emb_flat=q_mask_emb_flat,
            q_cls_flat=q_cls_flat,
            gt_labels_pad=gt_labels_pad,
            gt_masks_pad=gt_masks_pad,
            feature_maps_val=feature_maps_val,
            img_shape=(H_img, W_img),
            aggregation_similarity_metric=model.aggregation_similarity_metric,
        )
        
        loss_seed, loss_seed_sig = self._compute_seed_losses(
            q_sig_flat=q_sig_flat,
            gt_sigs_norm=gt_sigs_norm,
            gt_pad_mask=gt_pad_mask,
            q_seed_logits_flat=q_seed_logits_flat,
            features=features,
            identity_similarity_metric=model.identity_similarity_metric,
        )

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
            "loss_cls": loss_cls,
            "loss_mask_ce": loss_mask_ce,
            "loss_mask_iou": loss_mask_iou,
            "loss_mask_total": total_loss_mask,
            "loss_inter": loss_inter,
        }
        for level_name, loss_value in level_loss_mask_ce.items():
            components[f"loss_mask_ce_{level_name}"] = loss_value
        for level_name, loss_value in level_loss_mask_iou.items():
            components[f"loss_mask_iou_{level_name}"] = loss_value

        return final_loss, components


class StandardMask2FormerCriterion(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        return self.compute_loss(model, raw, targets)

    def _compute_single_layer_loss(
        self,
        feature_maps: list[tuple[str, torch.Tensor]],
        q_mask_emb: torch.Tensor,
        q_cls: torch.Tensor,
        targets,
        img_shape: tuple[int, int],
    ):
        num_classes = q_cls.shape[-1]
        reference_features = feature_maps[0][1]
        class_weights = torch.ones(num_classes, device=reference_features.device, dtype=reference_features.dtype)
        class_weights[0] = self.cfg.no_object_weight

        cls_losses = []
        matched_mask_logits = {level_name: [] for level_name, _ in feature_maps}
        matched_mask_targets = {level_name: [] for level_name, _ in feature_maps}

        for b in range(reference_features.shape[0]):
            labels = targets[b]["labels"].to(reference_features.device)
            masks = targets[b]["masks"].to(reference_features.device).float()
            fg_keep = labels != 0
            gt_labels = labels[fg_keep]
            gt_masks = masks[fg_keep]

            target_classes = torch.zeros(q_cls.shape[1], dtype=torch.long, device=reference_features.device)

            if gt_labels.numel() > 0:
                match_features = feature_maps[0][1]
                match_mask_logits = project_mask_embeddings(
                    q_mask_emb[b:b + 1],
                    match_features[b:b + 1],
                    img_shape,
                )[0]
                # Match the current layer queries [Q,...] against foreground GTs [M,...].
                matched_query_idx, matched_gt_idx = hungarian_mask2former_assignment(
                    cls_logits=q_cls[b],
                    mask_logits=match_mask_logits,
                    gt_labels=gt_labels,
                    gt_masks=gt_masks,
                    cfg=self.cfg,
                )
                target_classes[matched_query_idx] = gt_labels[matched_gt_idx]

                if matched_query_idx.numel() > 0:
                    for level_name, level_features in feature_maps:
                        level_mask_logits = project_mask_embeddings(
                            q_mask_emb[b:b + 1],
                            level_features[b:b + 1],
                            img_shape,
                        )[0]
                        matched_mask_logits[level_name].append(level_mask_logits[matched_query_idx])
                        matched_mask_targets[level_name].append(gt_masks[matched_gt_idx])

            cls_losses.append(F.cross_entropy(q_cls[b], target_classes, weight=class_weights))

        if cls_losses:
            loss_cls = torch.stack(cls_losses).mean()
        else:
            loss_cls = reference_features.sum() * 0.0

        level_loss_mask_bce = {}
        level_loss_mask_dice = {}
        zero = reference_features.sum() * 0.0
        for level_name, _ in feature_maps:
            if matched_mask_logits[level_name]:
                matched_logits = torch.cat(matched_mask_logits[level_name], dim=0)
                matched_targets = torch.cat(matched_mask_targets[level_name], dim=0)
                level_loss_mask_bce[level_name] = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
                level_loss_mask_dice[level_name] = dice_loss_from_logits(matched_logits, matched_targets).mean()
            else:
                level_loss_mask_bce[level_name] = zero
                level_loss_mask_dice[level_name] = zero

        loss_mask_bce = torch.stack(list(level_loss_mask_bce.values())).mean()
        loss_mask_dice = torch.stack(list(level_loss_mask_dice.values())).mean()

        return loss_cls, loss_mask_bce, loss_mask_dice, level_loss_mask_bce, level_loss_mask_dice

    def compute_loss(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        feature_maps = [
            (level_name, raw.feature_maps[level_name])
            for level_name in model.get_mask_loss_feature_levels()
            if level_name in raw.feature_maps
        ]
        reference_features = raw.features
        # raw.mask_embs/raw.cls_preds are lists over decoder layers.
        layer_losses = [
            self._compute_single_layer_loss(feature_maps, q_mask_emb, q_cls, targets, raw.img_shape)
            for q_mask_emb, q_cls in zip(raw.mask_embs, raw.cls_preds)
        ]

        loss_cls = torch.stack([loss[0] for loss in layer_losses]).mean()
        loss_mask_bce = torch.stack([loss[1] for loss in layer_losses]).mean()
        loss_mask_dice = torch.stack([loss[2] for loss in layer_losses]).mean()
        level_names = [level_name for level_name, _ in feature_maps]

        total_loss_mask = self.cfg.w_mask_bce * loss_mask_bce + self.cfg.w_mask_dice * loss_mask_dice
        final_loss = loss_cls + total_loss_mask
        zero = reference_features.sum() * 0.0

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
        for level_name in level_names:
            components[f"loss_mask_bce_{level_name}"] = torch.stack([loss[3][level_name] for loss in layer_losses]).mean()
            components[f"loss_mask_dice_{level_name}"] = torch.stack([loss[4][level_name] for loss in layer_losses]).mean()
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
