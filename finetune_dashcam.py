import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from learner import Learner
from dataset_dc import Normal_Loader, Anomaly_Loader
from sklearn import metrics
import os
import argparse
import numpy as np

# ----------------------------
# ARGUMENT PARSER
# ----------------------------
parser = argparse.ArgumentParser(description='Segment-level Supervised Fine-tuning for DashCam Anomaly Detection')
parser.add_argument('--dataset', default='./DashCam/', type=str, help='path to dataset')
parser.add_argument('--pretrained', default='./checkpoint/ckpt.pth', type=str, help='pretrained model path')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--w', default=1e-3, type=float, help='weight decay')
parser.add_argument('--modality', default='TWO', type=str, help='RGB | FLOW | TWO')
parser.add_argument('--input_dim', default=2048, type=int, help='input dimension (RGB=1024, FLOW=1024, TWO=2048)')
parser.add_argument('--drop', default=0.6, type=float, help='dropout rate')
parser.add_argument('--epochs', default=150, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=30, type=int, help='batch size per loader (anomaly/normal)')
args = parser.parse_args()

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# DATASET & LOADER
# ----------------------------
normal_train_dataset = Normal_Loader(is_train=1, path=args.dataset, modality=args.modality)
anomaly_train_dataset = Anomaly_Loader(is_train=1, path=args.dataset, modality=args.modality)
normal_test_dataset = Normal_Loader(is_train=0, path=args.dataset, modality=args.modality)
anomaly_test_dataset = Anomaly_Loader(is_train=0, path=args.dataset, modality=args.modality)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)

if os.path.exists(args.pretrained):
    checkpoint = torch.load(args.pretrained, map_location=device)
    state_dict = checkpoint.get('net', checkpoint)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    print("Loaded pretrained weights from", args.pretrained)
else:
    print("No pretrained weights found. Training from scratch.")

# ----------------------------
# OPTIMIZER & SCHEDULER
# ----------------------------
optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40])

# ----------------------------
# COMPUTE POS_WEIGHT
# ----------------------------
# Hitung jumlah total segmen positif/negatif dari dataset
total_pos, total_neg = 0, 0
for feats, lbls in anomaly_train_loader:
    total_pos += (lbls.numpy() == 1).sum()
    total_neg += (lbls.numpy() == 0).sum()
for feats, lbls in normal_train_loader:
    total_pos += (lbls.numpy() == 1).sum()
    total_neg += (lbls.numpy() == 0).sum()

pos_weight = total_neg / (total_pos + 1e-8)
print(f"Segment-level BCE pos_weight = {pos_weight:.4f}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

# ----------------------------
# HISTORY
# ----------------------------
best_auc = 0.0
train_loss_history = []
test_auc_history = []
test_acc_history = []

# ----------------------------
# TRAIN FUNCTION
# ----------------------------
def train(epoch):
    model.train()
    total_loss = 0.0
    iters = 0
    for (anom_feats, anom_labels), (norm_feats, norm_labels) in zip(anomaly_train_loader, normal_train_loader):
        # Move to device
        anom_feats = anom_feats.to(device).float()
        anom_labels = anom_labels.to(device).float()
        norm_feats = norm_feats.to(device).float()
        norm_labels = norm_labels.to(device).float()

        # Concatenate video batch
        feats = torch.cat([anom_feats, norm_feats], dim=0)
        labels = torch.cat([anom_labels, norm_labels], dim=0)

        batch_videos = feats.size(0)

        # Flatten segments
        inputs = feats.view(-1, feats.size(-1))
        inputs = F.normalize(inputs, p=2, dim=1)
        labels = labels.view(-1)

        # Forward
        outputs = model(inputs).squeeze()
        if outputs.dim() > 1:
            outputs = outputs.view(-1)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        iters += 1

    avg_loss = total_loss / (iters if iters > 0 else 1)
    train_loss_history.append(avg_loss)
    scheduler.step()
    print(f"Epoch {epoch+1}/{args.epochs} | Train Loss = {avg_loss:.6f}")
    return avg_loss

# ----------------------------
# TEST FUNCTION
# ----------------------------
# ----------------------------
# TEST FUNCTION (segment-level evaluation + debug)
# ----------------------------
def test_abnormal(epoch):
    global best_auc
    model.eval()
    all_scores = []
    all_labels = []
    video_lengths = []

    with torch.no_grad():
        # Anomaly videos
        for feats, labels in anomaly_test_loader:
            feats, labels = feats.to(device).float(), labels.to(device).float()
            inputs = feats.view(-1, feats.size(-1))
            inputs = F.normalize(inputs, p=2, dim=1)
            outputs = model(inputs).cpu().numpy().reshape(-1)[:labels.numel()]
            all_scores.append(outputs)
            all_labels.append(labels.cpu().numpy().reshape(-1))
            video_lengths.append(labels.numel())

        # Normal videos
        for feats, labels in normal_test_loader:
            feats, labels = feats.to(device).float(), labels.to(device).float()
            inputs = feats.view(-1, feats.size(-1))
            inputs = F.normalize(inputs, p=2, dim=1)
            outputs = model(inputs).cpu().numpy().reshape(-1)[:labels.numel()]
            all_scores.append(outputs)
            all_labels.append(labels.cpu().numpy().reshape(-1))
            video_lengths.append(labels.numel())

    # Flatten
    score_all = np.concatenate(all_scores, axis=0)
    gt_all = np.concatenate(all_labels, axis=0)

    # AUC & Accuracy
    if len(np.unique(gt_all)) > 1:
        fpr, tpr, _ = metrics.roc_curve(gt_all, score_all, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = 0.0

    preds = (score_all > 0.540).astype(int)
    accuracy = (preds == gt_all).mean()

    test_auc_history.append(auc)
    test_acc_history.append(accuracy)

    if auc > best_auc:
        best_auc = auc
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save({'net': model.state_dict()}, './checkpoint/ckpt_dashcam.pth')
        print(f"✅ Epoch {epoch+1} | Model Saved | Seg-AUC: {auc:.4f}, Seg-Acc: {accuracy:.4f}")
    else:
        print(f"Epoch {epoch+1} | Seg-AUC: {auc:.4f}, Seg-Acc: {accuracy:.4f}")

    # ================== 🔎 Diagnostik Anti-Flat ==================
    score_all_sig = 1 / (1 + np.exp(-score_all))
    idx_pos = (gt_all == 1)
    idx_neg = (gt_all == 0)

    mean_pos = score_all_sig[idx_pos].mean() if idx_pos.any() else float('nan')
    mean_neg = score_all_sig[idx_neg].mean() if idx_neg.any() else float('nan')
    prop_pred_pos_all = (score_all_sig > 0.540).mean()

    # split per-video
    def split_by_video(flat_arr, video_lengths):
        out = []
        st = 0
        for L in video_lengths:
            out.append(flat_arr[st:st+L])
            st += L
        return out

    vids_scores = split_by_video(score_all_sig, video_lengths)
    vids_labels = split_by_video(gt_all, video_lengths)

    prop_pred_per_video = [(v > 0.540).mean() for v in vids_scores]
    prop_gt_per_video   = [(v == 1).mean() for v in vids_labels]

    prop_pred_norm = [p for p, g in zip(prop_pred_per_video, prop_gt_per_video) if g == 0.0]
    prop_pred_anom = [p for p, g in zip(prop_pred_per_video, prop_gt_per_video) if g > 0.0]

    avg_prop_norm = np.mean(prop_pred_norm) if len(prop_pred_norm) > 0 else float('nan')
    avg_prop_anom = np.mean(prop_pred_anom) if len(prop_pred_anom) > 0 else float('nan')

    num_norm_all_one = sum(p >= 0.999 for p, g in zip(prop_pred_per_video, prop_gt_per_video) if g == 0.0)
    num_norm_videos  = sum(g == 0.0 for g in prop_gt_per_video)

    print("[Diag] mean(score|GT=1) = {:.3f} ; mean(score|GT=0) = {:.3f}".format(mean_pos, mean_neg))
    print("[Diag] prop(pred=1) overall = {:.3f}".format(prop_pred_pos_all))
    print("[Diag] avg prop(pred=1) per-video: NORMAL={:.3f} ; ANOMALY={:.3f}".format(avg_prop_norm, avg_prop_anom))
    print("[Diag] normal videos all-ones prediction: {}/{}".format(num_norm_all_one, num_norm_videos))

    idx_sorted = np.argsort(-np.array(prop_pred_per_video))
    print("[Diag] Top-3 videos by prop(pred=1):")
    for j in idx_sorted[:3]:
        print("  vid#{:d}: prop_pred={:.2f}, prop_gt={:.2f}".format(
            j, prop_pred_per_video[j], prop_gt_per_video[j]
        ))

    return auc


# ----------------------------
# MAIN LOOP
# ----------------------------
for epoch in range(args.epochs):
    train(epoch)
    test_abnormal(epoch)

print("\n===== FINAL SUMMARY =====")
for i in range(len(train_loss_history)):
    print(f"Epoch {i+1:02d} | Train Loss: {train_loss_history[i]:.6f} | "
          f"AUC: {test_auc_history[i]:.6f} | Acc: {test_acc_history[i]:.6f}")
print(f"Best Seg-AUC: {best_auc:.6f}")
