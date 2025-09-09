import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

# ----------------------------
# DEFINE LEARNER
# ----------------------------

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
        # manual linear path for legacy compatibility
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        return self.vars

# ----------------------------
# MAP SNIPPET SCORE KE FRAME
# ----------------------------
def map_scores_to_frames(score, n_frames):
    n_snippets = len(score)
    snippet_size = n_frames / n_snippets
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

# ----------------------------
# GET TOTAL FRAMES VIA FFMPEG
# ----------------------------
def get_total_frames(video_path):
    """Mengembalikan total frame video menggunakan ffmpeg/ffprobe"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error saat membaca video: {result.stderr}")
    return int(result.stdout.strip())

# ----------------------------
# INTERACTIVE INPUT
# ----------------------------
VIDEO_NPY = input("Masukkan nama file npy (misal ANOMALY/DashCam_00217.npy): ").strip()
VIDEO_FOLDER = VIDEO_NPY.split("/")[0]  # ANOMALY atau NORMAL
VIDEO_NAME = os.path.basename(VIDEO_NPY).replace("DashCam_", "")  # hapus prefix DashCam_
VIDEO_MP4 = os.path.join("..", "VIDEO", VIDEO_FOLDER, VIDEO_NAME.replace(".npy", ".mp4"))

# otomatis hitung total frame
N_FRAMES = get_total_frames(VIDEO_MP4)
print(f"Total frame video: {N_FRAMES}")

GT_START = int(input("Masukkan start frame anomaly: "))
GT_END   = int(input("Masukkan end frame anomaly: "))

RGB_PATH  = os.path.join("DashCamBest", "all_rgbs", VIDEO_FOLDER, os.path.basename(VIDEO_NPY))
FLOW_PATH = os.path.join("DashCamBest", "all_flows", VIDEO_FOLDER, os.path.basename(VIDEO_NPY))
CHECKPOINT = './checkpoint/finetuned_final_drop_0.6.pth'
INPUT_DIM = 2048
DROP_P = 0.6
THRESHOLD = 0.95
DEVICE = torch.device('cpu')

# ----------------------------
# LOAD MODEL
# ----------------------------
model = Learner(input_dim=INPUT_DIM, drop_p=DROP_P).to(DEVICE)
if os.path.exists(CHECKPOINT):
    ck = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ck['net'])
    print("Loaded checkpoint:", CHECKPOINT)
else:
    raise FileNotFoundError("Checkpoint tidak ditemukan!")
model.eval()

# ----------------------------
# LOAD FEATURES & PREDICTION
# ----------------------------
rgb_feats  = np.load(RGB_PATH)
flow_feats = np.load(FLOW_PATH)
video_feats = np.concatenate([rgb_feats, flow_feats], axis=1)
video_feats = torch.tensor(video_feats, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    outputs = model(video_feats)
    scores = 1 - outputs.squeeze().cpu().numpy()

frame_scores = map_scores_to_frames(scores, N_FRAMES)
status_per_frame = np.where(frame_scores >= THRESHOLD, "ANOMALY", "NORMAL")

# ----------------------------
# SAVE CSV & PLOT
# ----------------------------
output_dir = os.path.join(os.getcwd(), "..", "OUTPUT", VIDEO_NAME.replace(".npy",""))
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame({
    "frame": np.arange(1, N_FRAMES+1),
    "score": frame_scores,
    "threshold": THRESHOLD,
    "status": status_per_frame
})
csv_path = os.path.join(output_dir, f"{VIDEO_NAME.replace('.npy','')}_frame_scores.csv")
df.to_csv(csv_path, index=False)
print("Saved CSV:", csv_path)

plt.figure(figsize=(12,4))
plt.plot(range(1, N_FRAMES+1), frame_scores, label='Predicted Score', color='blue', marker='o')
plt.axvspan(GT_START, GT_END, color='lightgreen', alpha=0.3, label='Ground Truth Anomaly')
plt.axhline(THRESHOLD, color='r', linestyle='--', label='Threshold')
plt.xlabel("Frame")
plt.ylabel("Score")
plt.title(f"Anomaly Prediction vs Ground Truth - {VIDEO_NAME}")
plt.legend()
plt.savefig(os.path.join(output_dir, f"{VIDEO_NAME.replace('.npy','')}_plot.png"))
plt.close()

# ----------------------------
# VIDEO OVERLAY DENGAN OPENCV
# ----------------------------
cap = cv2.VideoCapture(VIDEO_MP4)
if not cap.isOpened():
    raise FileNotFoundError(f"Video tidak ditemukan atau tidak bisa dibaca: {VIDEO_MP4}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = os.path.join(output_dir, VIDEO_NAME.replace(".npy", "_overlay.mp4"))
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= N_FRAMES:
        break

    text = f"Score: {frame_scores[frame_idx]:.4f} | Threshold: {THRESHOLD:.4f} | {status_per_frame[frame_idx]}"
    
    # Ukuran font lebih besar
    font_scale = 1.2
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Background hitam semi-transparan untuk teks
    overlay = frame.copy()
    cv2.rectangle(overlay, (5,5), (5+text_w, 5+text_h+5), (0,0,0), -1)
    alpha = 0.6  # transparansi
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Tulis teks di atas background
    cv2.putText(frame, text, (10, 10 + text_h), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # Kotak merah untuk anomaly
    if status_per_frame[frame_idx] == "ANOMALY":
        cv2.rectangle(frame, (0,0), (width-1,height-1), (0,0,255), 10)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Saved overlay video:", out_path)
