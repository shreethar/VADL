import torch
import torch.nn.functional as F

def frame_scores_from_heatmap(heatmap, k_spatial=1):
    # heatmap: [B, T, 1, H, W] (logits)
    B, T, _, H, W = heatmap.shape
    flat = heatmap.view(B, T, -1)  # [B, T, H*W]
    if k_spatial <= 1:
        frame_scores = flat.max(dim=2).values  # [B,T]
    else:
        k_spatial = min(k_spatial, flat.size(2))
        topk, _ = torch.topk(flat, k=k_spatial, dim=2)
        frame_scores = topk.mean(dim=2)  # [B,T]
    return frame_scores  # logits

def video_scores_from_frame_scores(frame_scores, seq_lens, labels=None):
    # frame_scores: [B,T] (logits)
    B, T = frame_scores.shape
    video_scores = []
    # tolerate seq_lens as None, tensor or list
    if seq_lens is None:
        seq_lens_iter = [T] * B
    elif isinstance(seq_lens, torch.Tensor):
        seq_lens_iter = [int(x.item()) for x in seq_lens]
    else:
        seq_lens_iter = [int(x) for x in seq_lens]

    for i in range(B):
        valid_len = seq_lens_iter[i]
        if valid_len <= 0:
            valid_len = T
        valid = frame_scores[i, :valid_len]  # [valid_len]

        if labels is None:
            # testing / label-agnostic pooling
            k_time = max(1, valid_len // 16)
        else:
            lab = int(labels[i].item())
            if lab == 0:
                k_time = 1
            else:
                k_time = max(1, valid_len // 16 + 1)

        k_time = min(k_time, valid_len)
        topk_vals, _ = torch.topk(valid, k=k_time, largest=True)
        video_score = topk_vals.mean()
        video_scores.append(video_score)
    return torch.stack(video_scores, dim=0)  # [B]

def combined_criterion(
        video_labels, classifier_logits, heatmaps,
        seq_lens=None,
        class_weight=0.75, heatmap_weight=5.0,
        lambda_tv=0.2, lambda_temp=0.75, lambda_area=0.01,
        lambda_consistency=1.5,         # weight for MSE(prob,prob)
        lambda_frame_consistency=0.1,    # weight for BCE(logits, hm_prob) - new stronger consistency
        k_spatial=None,
        pos_weight=None):
    """
    classifier_logits: [B, T, 1] (logits)
    heatmaps: [B, T, 1, H, W] (logits)
    video_labels: [B] (0/1)
    seq_lens: None or tensor/list of lengths (len B). If None assumes full T.
    k_spatial: if None -> default = max(1, (H*W)//16 + 1)
    pos_weight: float or tensor passed to BCEWithLogitsLoss for video-level loss
    """
    B, T, _, H, W = heatmaps.shape
    device = heatmaps.device

    # --- decide k_spatial default ---
    total_patches = H * W
    if k_spatial is None:
        default_k = max(1, (total_patches // 16) + 1)   # analogous to temporal pooling rule
        k_spatial = default_k

    # --- heatmap -> frame scores (logits) ---
    frame_scores_hm = frame_scores_from_heatmap(heatmaps, k_spatial=k_spatial)  # [B,T], logits

    # --- classifier frame scores (logits) ---
    frame_scores_cls = classifier_logits.squeeze(-1)  # [B,T] logits

    # --- video scores (label-aware pooling during training if labels provided) ---
    video_score_hm = video_scores_from_frame_scores(frame_scores_hm, seq_lens=seq_lens or [T]*B, labels=video_labels)
    video_score_cls = video_scores_from_frame_scores(frame_scores_cls, seq_lens=seq_lens or [T]*B, labels=video_labels)

    # --- BCE loss function for video-level supervision (logits) ---
    if pos_weight is not None:
        if isinstance(pos_weight, (float, int)):
            pos_weight_t = torch.tensor([pos_weight], device=device)
        else:
            pos_weight_t = pos_weight.to(device)
        bce_video_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        bce_video_fn = torch.nn.BCEWithLogitsLoss()

    loss_heatmap = bce_video_fn(video_score_hm, video_labels.float())
    loss_classifier = bce_video_fn(video_score_cls, video_labels.float())

    # --- regularizers on heatmap probabilities ---
    prob_hm = torch.sigmoid(heatmaps)  # [B,T,1,H,W]

    # TV (spatial)
    h_diff = prob_hm[:, :, :, :, :-1] - prob_hm[:, :, :, :, 1:]
    w_diff = prob_hm[:, :, :, :-1, :] - prob_hm[:, :, :, 1:, :]
    tv_loss = (h_diff.abs().mean() + w_diff.abs().mean())

    # temporal consistency (frame-to-frame)
    if video_labels.sum() > 0:
        temp_loss = (prob_hm[:, 1:] - prob_hm[:, :-1]).abs().mean()
    else:
        temp_loss = torch.tensor(0, dtype = torch.int16)

    # area / sparsity (encourage small anomalous regions)
    area_loss = prob_hm.mean()

    # --- consistency between classifier frame probs and heatmap frame probs ---
    cls_probs = torch.sigmoid(frame_scores_cls)   # [B,T]
    hm_frame_probs = torch.sigmoid(frame_scores_hm)  # [B,T]

    # a) soft-prob MSE (existing)
    consistency_mse = F.mse_loss(cls_probs, hm_frame_probs)

    # b) stronger per-frame soft-label BCE: classifier logits -> hm_frame_probs (detached)
    #    encourages classifier logits to align with localization soft-labels (heatmap)
    bce_frame_fn = torch.nn.BCEWithLogitsLoss()
    # use hm_frame_probs.detach() to avoid double-backprop into heatmap head via this term
    consistency_frame_bce = bce_frame_fn(frame_scores_cls, hm_frame_probs.detach())

    # combine consistency terms (weights exposed above)
    consistency_total = lambda_consistency * consistency_mse + lambda_frame_consistency * consistency_frame_bce

    # total loss
    total = (heatmap_weight * loss_heatmap) + (class_weight * loss_classifier) \
            + lambda_tv * tv_loss + lambda_temp * temp_loss + lambda_area * area_loss \
            + consistency_total

    # assemble logs
    logs = {
        "loss_heatmap": float(loss_heatmap.detach().cpu().item()),
        "loss_classifier": float(loss_classifier.detach().cpu().item()),
        "tv_loss": float(tv_loss.detach().cpu().item()),
        "temp_loss": float(temp_loss.detach().cpu().item()),
        "area_loss": float(area_loss.detach().cpu().item()),
        "consistency_mse": float(consistency_mse.detach().cpu().item()),
        "consistency_frame_bce": float(consistency_frame_bce.detach().cpu().item()),
        "consistency_total": float(consistency_total.detach().cpu().item()),
        "video_score_hm_mean": float(video_score_hm.detach().mean().cpu().item()),
        "video_score_cls_mean": float(video_score_cls.detach().mean().cpu().item()),
        "k_spatial_used": int(k_spatial)
    }

    return total, logs
