import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, is_transformer=0, device="cpu"):
    # initialize loss values on the correct device
    loss = torch.tensor(0., device=device)
    loss_intra = torch.tensor(0., device=device)
    sparsity = torch.tensor(0., device=device)
    smooth = torch.tensor(0., device=device)

    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(30, device=device)
        normal_index = torch.randperm(30, device=device)

        y_anomaly = y_pred[i, :32][anomaly_index]
        y_normal  = y_pred[i, 32:][normal_index]

        y_anomaly_max = torch.max(y_anomaly)
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1. - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2) * 0.00008

    loss = (loss + sparsity + smooth) / batch_size

    return loss