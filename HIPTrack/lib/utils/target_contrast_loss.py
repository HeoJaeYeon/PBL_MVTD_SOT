import torch
import torch.nn.functional as F


def box_to_token_masks(box_xywh, feat_sz):
    """Assign the tokens of a feat_sz x feat_sz grid to the target box.

    args:
        box_xywh - (B, 4) target box (x1, y1, w, h), normalized to [0, 1] in crop coordinates
        feat_sz - number of tokens along one side of the grid
    returns:
        pos - (B, feat_sz*feat_sz) bool, token center inside the box (+ the token holding the box center)
        neg - (B, feat_sz*feat_sz) bool, token cell does not overlap the box at all
        Tokens that are neither (cells cut by the box border) are ignored.
    """
    B = box_xywh.shape[0]
    device = box_xywh.device
    idx = torch.arange(feat_sz, device=device, dtype=box_xywh.dtype)
    cell_y1, cell_x1 = torch.meshgrid(idx / feat_sz, idx / feat_sz, indexing='ij')
    cell_x1, cell_y1 = cell_x1.reshape(1, -1), cell_y1.reshape(1, -1)
    cell_x2, cell_y2 = cell_x1 + 1.0 / feat_sz, cell_y1 + 1.0 / feat_sz
    ctr_x, ctr_y = cell_x1 + 0.5 / feat_sz, cell_y1 + 0.5 / feat_sz

    x1, y1, w, h = [v[:, None] for v in box_xywh.unbind(-1)]
    x2, y2 = x1 + w, y1 + h

    pos = (ctr_x >= x1) & (ctr_x <= x2) & (ctr_y >= y1) & (ctr_y <= y2)
    # small or thin boxes may contain no token center: the token holding the box center is always positive
    bx = ((x1 + w / 2).squeeze(1) * feat_sz).floor().long().clamp(0, feat_sz - 1)
    by = ((y1 + h / 2).squeeze(1) * feat_sz).floor().long().clamp(0, feat_sz - 1)
    pos[torch.arange(B, device=device), by * feat_sz + bx] = True

    overlap = (cell_x2 > x1) & (cell_x1 < x2) & (cell_y2 > y1) & (cell_y1 < y2)
    neg = ~overlap & ~pos
    return pos, neg


def target_contrast_loss(feat, box_xywh, tau=0.1):
    """Target-background contrastive loss (multi-positive InfoNCE against the target prototype).

    The prototype is the mean feature of the target tokens. Target tokens are pulled towards it and
    background tokens (distractor vessels, waves, glare, ...) are pushed away from it:
        L = -log( sum_pos exp(cos(p, f_i) / tau) / sum_{pos+neg} exp(cos(p, f_k) / tau) )

    args:
        feat - (B, C, H, W) search-region feature fed to the box head (must be produced by trainable layers)
        box_xywh - (B, 4) target box (x1, y1, w, h), normalized in search-crop coordinates
        tau - temperature
    returns:
        scalar loss averaged over the samples that have at least one positive and one negative token
    """
    B, C, H, W = feat.shape
    assert H == W, "square feature map expected"
    assert box_xywh.shape[0] == B, "feature / box batch mismatch: {} vs {}".format(B, box_xywh.shape[0])

    tokens = F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)          # (B, HW, C)
    pos, neg = box_to_token_masks(box_xywh.to(tokens.dtype), H)

    prototype = F.normalize((tokens * pos.unsqueeze(-1)).sum(1) / pos.sum(1, keepdim=True), dim=-1)  # (B, C)
    logits = torch.einsum('bc,blc->bl', prototype, tokens) / tau            # (B, HW)

    min_val = torch.finfo(logits.dtype).min
    log_all = torch.logsumexp(logits.masked_fill(~(pos | neg), min_val), dim=1)
    log_pos = torch.logsumexp(logits.masked_fill(~pos, min_val), dim=1)
    loss = log_all - log_pos                                               # >= 0

    keep = pos.any(1) & neg.any(1)
    if keep.any():
        return loss[keep].mean()
    return feat.sum() * 0.0
