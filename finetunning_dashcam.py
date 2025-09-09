#!/usr/bin/env python3
"""
finetunning_dashcam_fixed.py

CPU-only fine-tuning script for DashCam anomaly dataset.

Assumptions:
- Each .npy feature file contains per-segment features with shape (n_segments, feat_dim).
  Typical case: (32, feat_dim).
- Train/test .txt formats:
  - train_normal.txt / train_anomaly.txt: one path per line (relative basename, e.g. Video001 or Folder/Video001.mp4)
  - test_normalv2.txt: "<path> <n_frames> -1"
  - test_anomalyv2.txt: "<path>|<n_frames>|[s, e]"   (can be multiple ranges separated by comma)
- Directory layout (example):
  DashCam/
    all_rgbs/
      ANOMALY/
      NORMAL/
    all_flows/
      ANOMALY/
      NORMAL/
    train_normal.txt
    train_anomaly.txt
    test_normalv2.txt
    test_anomalyv2.txt
"""

import os
import random
import argparse
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------
# Utility functions
# ---------------------------
def ensure_npy_name(name: str) -> str:
    """
    Return a filename ending with '.npy' matching typical filenames on disk.
    If the .txt contains 'Folder/Video.mp4' or 'Video.mp4', convert to 'Video.npy'.
    If .txt already gives 'Video.npy' or 'Video.mp4.npy', handle gracefully.
    IMPORTANT: adjust if your disk actually contains 'video.mp4.npy' style names.
    """
    base = os.path.basename(name)
    # remove possible trailing '.mp4.npy' -> convert to '.npy'
    if base.endswith('.mp4.npy'):
        return base  # keep as is if files indeed have .mp4.npy
    # if endswith .npy, return
    if base.endswith('.npy'):
        return base
    # if endswith .mp4, replace with .npy
    if base.endswith('.mp4'):
        return base[:-4] + '.npy'
    # else assume it's a basename (no extension) -> append .npy
    return base + '.npy'

def clean_name_prefix(name: str) -> str:
    """
    Remove common prefix folders used in some txt files, like 'Normal/', 'ANOMALY/', etc.
    """
    return name.replace('Normal/', '').replace('NORMAL/', '').replace('Anomaly/', '').replace('ANOMALY/', '')

# ---------------------------
# Dataset Loaders
# ---------------------------
class Normal_Loader(Dataset):
    """
    Loader for normal videos.
    When is_train==1 -> read train_normal.txt entries (one path per line)
    When is_train==0 -> read test_normalv2.txt (path frames -1)
    """
    def __init__(self, is_train=1, path='./DashCamBest/', modality='TWO'):
        super().__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.dir_rgbs = os.path.join(self.path, 'all_rgbs', 'NORMAL')
        self.dir_flows = os.path.join(self.path, 'all_flows', 'NORMAL')

        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = [l.strip() for l in f if l.strip()]
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = [l.strip() for l in f if l.strip()]
            random.shuffle(self.data_list)
            # keep behavior similar to original (optional): drop last 10 if many
            if len(self.data_list) > 10:
                self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        if self.is_train == 1:
            name = clean_name_prefix(entry)
            fname = ensure_npy_name(name)
            rgb_path = os.path.join(self.dir_rgbs, fname)
            flow_path = os.path.join(self.dir_flows, fname)
            rgb_npy = np.load(rgb_path)
            flow_npy = np.load(flow_path)
            concat = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return torch.from_numpy(rgb_npy).float()
            elif self.modality == 'FLOW':
                return torch.from_numpy(flow_npy).float()
            else:
                return torch.from_numpy(concat).float()
        else:
            # test normal: "path frames -1"
            parts = entry.split()
            name = clean_name_prefix(parts[0])
            frames = int(parts[1])
            # gts = -1 indicates no anomaly
            gts = -1
            fname = ensure_npy_name(name)
            rgb_npy = np.load(os.path.join(self.dir_rgbs, fname))
            flow_npy = np.load(os.path.join(self.dir_flows, fname))
            concat = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return torch.from_numpy(rgb_npy).float(), gts, frames
            elif self.modality == 'FLOW':
                return torch.from_numpy(flow_npy).float(), gts, frames
            else:
                return torch.from_numpy(concat).float(), gts, frames

class Anomaly_Loader(Dataset):
    """
    Loader for anomaly videos.
    When is_train==1 -> read train_anomaly.txt
    When is_train==0 -> read test_anomalyv2.txt (path|frames|[s,e])
    """
    def __init__(self, is_train=1, path='./DashCamBest/', modality='TWO'):
        super().__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.dir_rgbs = os.path.join(self.path, 'all_rgbs', 'ANOMALY')
        self.dir_flows = os.path.join(self.path, 'all_flows', 'ANOMALY')

        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = [l.strip() for l in f if l.strip()]
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        if self.is_train == 1:
            name = clean_name_prefix(entry)
            fname = ensure_npy_name(name)
            rgb_npy = np.load(os.path.join(self.dir_rgbs, fname))
            flow_npy = np.load(os.path.join(self.dir_flows, fname))
            concat = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return torch.from_numpy(rgb_npy).float()
            elif self.modality == 'FLOW':
                return torch.from_numpy(flow_npy).float()
            else:
                return torch.from_numpy(concat).float()
        else:
            # test anomaly: "path|frames|[s,e]" (possibly spaces around |)
            parts = entry.split('|')
            name = clean_name_prefix(parts[0].strip())
            frames = int(parts[1].strip())
            gts_str = parts[2].strip()
            # remove [ and ] if present
            if gts_str.startswith('[') and gts_str.endswith(']'):
                gts_str = gts_str[1:-1]
            # split by comma -> get list of integers (could be multiple ranges)
            gts_parts = [p.strip() for p in gts_str.split(',') if p.strip()]
            gts = [int(x) for x in gts_parts] if len(gts_parts) > 0 else []
            fname = ensure_npy_name(name)
            rgb_npy = np.load(os.path.join(self.dir_rgbs, fname))
            flow_npy = np.load(os.path.join(self.dir_flows, fname))
            concat = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return torch.from_numpy(rgb_npy).float(), gts, frames
            elif self.modality == 'FLOW':
                return torch.from_numpy(flow_npy).float(), gts, frames
            else:
                return torch.from_numpy(concat).float(), gts, frames

# ---------------------------
# Models
# ---------------------------
# learner_fix.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = drop_p
        self.weight_init()
        # store parameters as ParameterList to mimic original code (not strictly necessary)
        self.vars = nn.ParameterList()
        for param in self.classifier.parameters():
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            return self.classifier(x)
        # legacy manual linear path (kept for compatibility)
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        return self.vars

class Learner2(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super().__init__()
        self.filter1 = nn.LayerNorm(input_dim)
        self.filter2 = nn.LayerNorm(input_dim)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(drop_p)

        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 256)
        self.fc4 = nn.Linear(256, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.relu2(x)
        out = self.fc5(self.relu(self.fc4((self.relu(self.fc3(self.dropout(self.relu(self.fc2(self.relu(self.fc1(x)))))))))))
        out2 = self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.dropout(self.relu(self.fc2(self.relu(self.fc1(x1))))))))))
        out = out + out2 * 0.2
        return torch.sigmoid(out)

# ---------------------------
# Loss: MIL (device-safe)
# ---------------------------
def MIL22(y_pred, batch_size, is_transformer=0):
    """
    y_pred shape expected: (batch_size * T, 1) or (batch_size, T) depending on forward.
    batch_size = num videos in the (concatenated) batch (anomaly+normal)
    This MIL is adapted from original code; it samples k indices to compute hinge loss between 
    max anomaly and max normal instance within each video pair in batch.
    """
    device = y_pred.device

    # Flatten to 1D if shape is (N,1)
    if y_pred.dim() == 2 and y_pred.size(1) == 1:
        y_pred = y_pred.view(-1)

    # Try to reshape to (batch_size, -1)
    try:
        y_pred = y_pred.view(batch_size, -1)
    except Exception:
        # if cannot reshape, return zero loss (safe fallback)
        return torch.tensor(0., device=device)

    loss = torch.tensor(0., device=device)
    sparsity = torch.tensor(0., device=device)
    smooth = torch.tensor(0., device=device)

    B, T = y_pred.size()

    # assume first half of snippets correspond to anomaly videos and second half to normal videos,
    # or if concatenated along batch dim, we consider T as number of snippets per video.
    half = T // 2 if T >= 2 else T

    # choose k samples (cap at half)
    k = min(30, max(1, half))

    for i in range(B):
        if half <= 0:
            continue

        anomaly_index = torch.randperm(half, device=device)[:k]
        normal_index = torch.randperm(max(1, T - half), device=device)[:k]

        y_anomaly = y_pred[i, :half][anomaly_index]
        y_normal = y_pred[i, half:half + (T - half)][normal_index]

        y_anomaly_max = torch.max(y_anomaly)
        y_normal_max = torch.max(y_normal)

        loss = loss + F.relu(1.0 - y_anomaly_max + y_normal_max)
        sparsity = sparsity + torch.sum(y_pred[i, :half]) * 0.00008

        if half > 1:
            smooth = smooth + torch.sum((y_pred[i, :half - 1] - y_pred[i, 1:half])**2) * 0.00008

    total = (loss + sparsity + smooth) / max(1, B)
    return total

def MIL(y_pred, batch_size, is_transformer=0):
    """
    Revised MIL loss function with improved efficiency and robustness.

    Args:
        y_pred (torch.Tensor): A tensor of predicted scores.
                                Expected shapes: (N_snippets, 1) or (N_snippets,) where N_snippets
                                is the total number of snippets in the batch.
        batch_size (int): The number of videos in the combined batch (anomaly + normal).
        is_transformer (int): Placeholder for future use (not used in this version).

    Returns:
        torch.Tensor: The total calculated MIL loss.
    """
    device = y_pred.device

    # Handle input shape variations
    if y_pred.dim() == 2 and y_pred.size(1) == 1:
        y_pred = y_pred.view(-1)
    
    # Reshape to (batch_size, -1)
    try:
        y_pred = y_pred.view(batch_size, -1)
    except RuntimeError:
        # Fallback for incompatible shapes, returning zero loss.
        return torch.tensor(0., device=device)

    # Ensure batch size is even and positive
    if batch_size <= 1 or batch_size % 2 != 0:
        return torch.tensor(0., device=device)

    B, T = y_pred.size()
    half = T // 2

    # Vectorized Hinge Loss
    # Find max scores for anomaly and normal videos
    #y_anomaly_max = torch.max(y_pred[:, :half], dim=1)[0]
    #y_normal_max = torch.max(y_pred[:, half:], dim=1)[0]
    
    # y_pred shape: (B, 32)

    # Mask untuk anomaly snippets: 1-10 (0-based: 0-9)
    anomaly_mask = torch.zeros_like(y_pred, dtype=torch.bool)
    anomaly_mask[:, 0:15] = True

    # Mask untuk normal snippets: 11-32 (0-based: 10-31)
    normal_mask = torch.zeros_like(y_pred, dtype=torch.bool)
    normal_mask[:, 15:] = True

    # Ambil max score masing-masing
    y_anomaly_max = torch.max(y_pred[anomaly_mask].view(y_pred.size(0), -1), dim=1)[0]
    y_normal_max = torch.max(y_pred[normal_mask].view(y_pred.size(0), -1), dim=1)[0]

    # Calculate hinge loss for all video pairs in the batch
    hinge_loss = F.relu(1.0 - y_anomaly_max + y_normal_max)
    loss = torch.mean(hinge_loss)

    # Vectorized Sparsity Loss
    # Sum scores for all anomaly snippets, then average across the batch
    sparsity = torch.mean(torch.sum(y_pred[:, :half], dim=1)) * 0.00008

    # Vectorized Smoothness Loss
    # Calculate difference between adjacent snippets, square, sum, and average
    smooth = torch.tensor(0., device=device)
    if half > 1:
        diffs = y_pred[:, :half-1] - y_pred[:, 1:half]
        smooth = torch.mean(torch.sum(diffs**2, dim=1)) * 0.00008

    total = loss + sparsity + smooth
    return total

# ---------------------------
# Utils: evaluation mapping
# ---------------------------
def map_scores_to_frames(score, n_frames):
    """
    Map per-snippet scores (numpy array shape (n_snippets,)) to frame-level scores length n_frames.
    snippet_size dihitung otomatis agar sesuai jumlah frame sebenarnya.
    """
    import numpy as np

    score = np.asarray(score).reshape(-1)
    n_snippets = score.shape[0]
    if n_frames <= 0 or n_snippets == 0:
        return np.zeros(n_frames, dtype=float)

    # Hitung snippet size secara dinamis sebagai float
    snippet_size = float(n_frames) / n_snippets
    frame_scores = np.zeros(n_frames, dtype=float)

    for j in range(n_snippets):
        start = int(round(j * snippet_size))
        end = int(round((j + 1) * snippet_size))
        start = max(0, min(start, n_frames))
        end = max(0, min(end, n_frames))
        if start >= end:
            if start < n_frames:
                frame_scores[start] = score[j]
        else:
            frame_scores[start:end] = score[j]

    return frame_scores

# ---------------------------
# Training & evaluation
# ---------------------------
def train_one_epoch(model, normal_loader, anomaly_loader, optimizer, criterion, device):
    """
    normal_loader and anomaly_loader must have same batch_size and drop_last=True to avoid mismatches.
    We zip(normal_loader, anomaly_loader) -> get pairs (normal_batch, anomaly_batch).
    Each batch returned is tensor shape: (B, n_snippets, feat_dim)
    We concatenate on batch axis (dim=0).
    """
    model.train()
    running_loss = 0.0
    iters = 0
    for normal_batch, anomaly_batch in zip(normal_loader, anomaly_loader):
        # normal_batch: either tensor (B, S, D) for train entries
        # anomaly_batch: same
        normal_inputs = normal_batch
        anomaly_inputs = anomaly_batch

        # If DataLoader yields tuples (e.g., (tensor, ..) in test mode), handle that
        if isinstance(normal_inputs, (list, tuple)):
            normal_inputs = normal_inputs[0]
        if isinstance(anomaly_inputs, (list, tuple)):
            anomaly_inputs = anomaly_inputs[0]

        # ensure tensors
        normal_inputs = normal_inputs.to(device)
        anomaly_inputs = anomaly_inputs.to(device)

        # print(f"[DEBUG] Batch {iters}: normal {normal_inputs.shape[1]} snippets, anomaly {anomaly_inputs.shape[1]} snippets")

        # concatenate on batch dimension
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)  # shape (2B, S, D)

        batch_size = inputs.shape[0]  # number of videos in this combined batch (2B)
        inputs_flat = inputs.view(-1, inputs.size(-1))  # (2B*S, D)

        outputs = model(inputs_flat)  # expected (2B*S, 1)
        loss = criterion(outputs, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iters += 1

    avg_loss = running_loss / max(1, iters)
    return avg_loss

from sklearn import metrics

def evaluate_auc(model, anomaly_test_loader, normal_test_loader, device, threshold=0.8933, plot_hist=False):
    model.eval()
    all_scores = []
    all_labels = []

    # -----------------------------
    # 1) DEBUG jumlah sample dataset
    # -----------------------------
    print(f"[INFO] Jumlah batch anomaly test : {len(anomaly_test_loader)}")
    print(f"[INFO] Jumlah batch normal  test : {len(normal_test_loader)}")

    with torch.no_grad():
        for (an_data, norm_data) in zip(anomaly_test_loader, normal_test_loader):
            # ----- anomaly video -----
            inputs, gts, frames = an_data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            scores = 1 - model(inputs).detach().cpu().numpy()   # dibalik (asumsi model prediksi normal tinggi)
            score_list = map_scores_to_frames(scores, int(frames[0]))


            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1

            # ----- normal video -----
            inputs2, gts2, frames2 = norm_data
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            scores2 = 1 - model(inputs2).detach().cpu().numpy()
            score_list2 = map_scores_to_frames(scores2, int(frames2[0]))

            gt_list2 = np.zeros(frames2[0])

            # gabungkan anomaly+normal
            scores_concat = np.concatenate([score_list, score_list2])
            gts_concat = np.concatenate([gt_list, gt_list2])

            all_scores.append(scores_concat)
            all_labels.append(gts_concat)

    # gabungkan semua video
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # -----------------------------
    # 2) DEBUG distribusi skor
    # -----------------------------
    print("[DEBUG] Score stats (ALL) -> min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
        all_scores.min(), all_scores.max(), all_scores.mean()
    ))

    anomaly_scores = all_scores[all_labels == 1]
    normal_scores  = all_scores[all_labels == 0]

    print("[DEBUG] Anomaly scores -> min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
        anomaly_scores.min() if len(anomaly_scores) else 0,
        anomaly_scores.max() if len(anomaly_scores) else 0,
        anomaly_scores.mean() if len(anomaly_scores) else 0,
    ))
    print("[DEBUG] Normal scores  -> min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
        normal_scores.min() if len(normal_scores) else 0,
        normal_scores.max() if len(normal_scores) else 0,
        normal_scores.mean() if len(normal_scores) else 0,
    ))

    # -----------------------------
    # 3) ROC & AUC (asli vs dibalik)
    # -----------------------------
    auc_raw = metrics.roc_auc_score(all_labels, all_scores)
    auc_flip = metrics.roc_auc_score(all_labels, 1 - all_scores)

    print(f"[DEBUG] AUC (raw)  = {auc_raw:.4f}")
    print(f"[DEBUG] AUC (1-x) = {auc_flip:.4f}")

    auc_value = auc_raw  # default pakai raw (kalau ternyata flip lebih bagus bisa dipakai)

    # -----------------------------
    # 3b) Cari threshold terbaik (Youden’s J)
    # -----------------------------
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores)
    j_scores = tpr - fpr
    j_best_idx = np.argmax(j_scores)
    best_threshold = thresholds[j_best_idx]
    print(f"[DEBUG] Best threshold (Youden J) = {best_threshold:.4f} | TPR={tpr[j_best_idx]:.4f}, FPR={fpr[j_best_idx]:.4f}")

    # threshold manual dari argumen (default=0.2)
    preds = (all_scores >= threshold).astype(int)

    # -----------------------------
    # 4) Confusion matrix & metrik
    # -----------------------------
    tn, fp, fn, tp = metrics.confusion_matrix(all_labels, preds).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)

    print(f"[DEBUG] Confusion Matrix (threshold={threshold}): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"[DEBUG] Metrics -> Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # -----------------------------
    # 5) Visualisasi distribusi skor
    # -----------------------------
    if plot_hist:
        plt.figure(figsize=(8, 5))
        plt.hist(normal_scores, bins=50, alpha=0.6, label="Normal", color="blue")
        plt.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", color="red")
        plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold}")
        plt.axvline(best_threshold, color="green", linestyle="--", label=f"Best Thr={best_threshold:.2f}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Scores (Normal vs Anomaly)")
        plt.legend()
        plt.show()

    return auc_value, best_threshold

def make_balanced_loader(dataset, batch_size=30, drop_last=True, num_workers=0):
    # 1) Tentukan target tiap video
    targets = [1 if "ANOMALY" in p else 0 for p in dataset.data_list]
    
    # 2) Hitung jumlah video per kelas
    class_sample_count = np.array([
        len(np.where(np.array(targets) == t)[0]) 
        for t in np.unique(targets)
    ])
    print("[DEBUG] Original class counts:")
    for t, c in zip(np.unique(targets), class_sample_count):
        print(f"  Class {t}: {c} videos")
    
    # 3) Hitung bobot
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])

    # 4) Buat sampler dan loader
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last, num_workers=num_workers)
    
    # 5) Debug: cek distribusi batch pertama
    first_batch = next(iter(loader))
    if isinstance(first_batch, torch.Tensor):
        batch_size_actual = first_batch.size(0)
    else:
        batch_size_actual = len(first_batch)
    print(f"[DEBUG] Total samples for WeightedRandomSampler: {len(samples_weight)}")
    print(f"[DEBUG] Example batch size: {batch_size_actual}")
    
    return loader


# ---------------------------
# Main finetune loop
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune anomaly detector (CPU) on DashCam dataset')
    parser.add_argument('--data_root', type=str, default='./DashCamBest/', help='root path to DashCam dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/ckpt.pth', help='pretrained checkpoint to load')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/', help='directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default= 100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per loader (use small on CPU)')
    parser.add_argument('--modality', type=str, default='TWO', choices=['RGB','FLOW','TWO'])
    parser.add_argument('--input_dim', type=int, default=2048, help='feature dim per segment (if TWO => combined dim)')
    parser.add_argument('--drop', type=float, default=0.6)
    parser.add_argument('--use_ffc', action='store_true', help='use Learner2 architecture')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    print('Using device:', device)

    # instantiate datasets/loaders
    train_normal = Normal_Loader(is_train=1, path=args.data_root, modality=args.modality)
    train_anomaly = Anomaly_Loader(is_train=1, path=args.data_root, modality=args.modality)

    print("All normal videos:", len(train_normal.data_list))
    print("All anomaly videos:", len(train_anomaly.data_list))

    test_normal = Normal_Loader(is_train=0, path=args.data_root, modality=args.modality)
    test_anomaly = Anomaly_Loader(is_train=0, path=args.data_root, modality=args.modality)

    # DataLoader: set drop_last=True for training loaders to avoid size mismatch on zip
    train_normal_loader = make_balanced_loader(train_normal, batch_size=args.batch_size, drop_last=True, num_workers=0)
    train_anomaly_loader = make_balanced_loader(train_anomaly, batch_size=args.batch_size, drop_last=True, num_workers=0)
    test_normal_loader = DataLoader(test_normal, batch_size=1, shuffle=False, num_workers=0)
    test_anomaly_loader = DataLoader(test_anomaly, batch_size=1, shuffle=False, num_workers=0)

    # model
    if args.use_ffc:
        model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
    else:
        model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)

    # load pretrained if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        print('Loading pretrained checkpoint:', args.checkpoint)
        ck = torch.load(args.checkpoint, map_location=device)
        if isinstance(ck, dict) and 'net' in ck:
            state = ck['net']
        elif isinstance(ck, dict) and 'state_dict' in ck:
            state = ck['state_dict']
        else:
            state = ck
        try:
            model.load_state_dict(state, strict=False)
            print('Loaded pretrained weights (strict=False).')
        except Exception as e:
            print('Warning: failed to fully load checkpoint:', e)
    else:
        print('No checkpoint found at', args.checkpoint, '-> training from scratch.')

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.8)], gamma=0.1)
    criterion = MIL

    best_auc = 0.0
    best_epoch = -1
    best_info = None
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_normal_loader, train_anomaly_loader, optimizer, criterion, device)
        print('Epoch [{}/{}] train_loss: {:.6f}'.format(epoch, args.epochs, avg_loss))
        scheduler.step()

        auc, best_threshold = evaluate_auc(model, test_anomaly_loader, test_normal_loader, device)
        print('Epoch [{}/{}] val AUC: {:.6f}'.format(epoch, args.epochs, auc))

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_info = {
                "train_loss": avg_loss,
                "auc": auc,
                "threshold": best_threshold,
            }
            save_path = os.path.join(args.save_dir, 'finetuned_best.pth')
            torch.save({'net': model.state_dict(), 'auc': best_auc, 'epoch': epoch}, save_path)
            print('Saved best checkpoint to', save_path)

    final_path = os.path.join(args.save_dir, 'finetuned_final.pth')
    torch.save({'net': model.state_dict(), 'auc': best_auc}, final_path)
    print('Finished. Best AUC: {:.6f}. Final model saved to {}'.format(best_auc, final_path))

    # 🔹 Tambahan recap best epoch
    if best_info is not None:
        print("="*60)
        print(">>> DETAIL BEST EPOCH <<<")
        print("Best Epoch: {}/{}".format(best_epoch, args.epochs))
        print("Train Loss: {:.6f}".format(best_info["train_loss"]))
        print("Best AUC  : {:.6f}".format(best_info["auc"]))
        print("Best Thr. : {:.4f}".format(best_info["threshold"]))
        print("="*60)

if __name__ == '__main__':
    main()
