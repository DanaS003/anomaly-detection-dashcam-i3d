import torch
import torch.nn.functional as F

def segment_level_loss(outputs, labels, pos_weight=None):
    """
    Hitung BCE loss antara prediksi segmen (32,) dan label segmen (32,).

    outputs : tensor (32,) logits prediksi segmen
    labels  : tensor (32,) biner {0,1}
    pos_weight : tensor [1], bobot untuk positive class
    """
    outputs = outputs.view(-1)   # (32,)
    labels = labels.view(-1)     # (32,)
    return F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight)


def mil_ranking_loss(outputs, batch_size):
    """
    Multiple Instance Learning (MIL) ranking loss.
    outputs: (B*32,) logits segmen
    batch_size: jumlah video dalam batch
    
    Asumsi separuh batch = anomaly, separuh = normal.
    """
    outputs = outputs.view(batch_size, -1)  # (B, 32)
    scores_max, _ = outputs.max(dim=1)      # (B,)

    scores_anom = scores_max[:batch_size // 2]
    scores_norm = scores_max[batch_size // 2:]

    # ranking loss: anomaly > normal
    loss = torch.relu(1.0 - (scores_anom.mean() - scores_norm.mean()))
    return loss


def sparsity_loss(outputs):
    """
    Sparsity loss untuk regularisasi output anomaly.
    outputs: (N,) logits segmen
    """
    return torch.mean(torch.sigmoid(outputs))


def hybrid_loss(outputs, batch_size, labels=None, device="cpu", pos_weight=None):
    """
    Gabungan MIL loss + sparsity + segment-level loss (kalau ada label).
    
    outputs: (B*32,) logits
    labels : (B, 32) tensor biner atau None
    pos_weight : float atau tensor (skalar)
    """
    loss_mil = mil_ranking_loss(outputs, batch_size)
    loss_sparse = sparsity_loss(outputs)

    if labels is not None:
        # Segment-level loss
        labels = labels.to(device).view(-1)   # (B*32,)
        outputs = outputs.view(-1)            # (B*32,)

        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor([pos_weight], device=device)

        loss_seg = F.binary_cross_entropy_with_logits(
            outputs, labels, pos_weight=pos_weight
        )
    else:
        loss_seg = torch.tensor(0.0, device=device)

    return loss_mil + 0.0005 * loss_sparse + 1.0 * loss_seg