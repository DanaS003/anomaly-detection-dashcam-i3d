import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import matplotlib.pyplot as plt


class Normal_Loader(Dataset):
    """ 
    Loader untuk video normal 
    is_train = 1 -> train, 0 -> test
    modality = 'RGB' / 'FLOW' / 'TWO'
    """
    def __init__(self, is_train=1, path='./DashCam/', modality='TWO'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path

        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')

        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

        if not self.is_train:
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx].strip().split()
        video_name = line[0]   # ex: Normal/DashCam_00990.mp4
        # txt format test_normal: "Normal/DashCam_xxxxx.mp4 numFrames -1"
        # txt format train_normal: "Normal/DashCam_xxxxx.mp4"

        # load fitur segmen-level
        rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', video_name + '.npy'))   # (32, 1024)
        flow_npy = np.load(os.path.join(self.path, 'all_flows', video_name + '.npy')) # (32, 1024)
        concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)                      # (32, 2048)

        # label semua segmen = 0
        labels = np.zeros(rgb_npy.shape[0], dtype=np.float32)  # (32,)

        if self.modality == 'RGB':
            return torch.from_numpy(rgb_npy).float(), torch.from_numpy(labels).float()
        elif self.modality == 'FLOW':
            return torch.from_numpy(flow_npy).float(), torch.from_numpy(labels).float()
        else:
            return torch.from_numpy(concat_npy).float(), torch.from_numpy(labels).float()


class Anomaly_Loader(Dataset):
    """ 
    Loader untuk video anomalous
    is_train = 1 -> train, 0 -> test
    Format file txt: videoName|numFrames|[start1,end1,start2,end2,...]
    """
    def __init__(self, is_train=1, path='./DashCam/', modality='TWO'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path

        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')

        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        parts = self.data_list[idx].strip().split('|')
        video_name = parts[0]                       # ex: Anomaly/DashCam_00903.mp4
        num_frames = int(parts[1])                  # total frames
        gts_str = parts[2].strip()
        gts = [int(i) for i in gts_str.strip("[]").split(',') if i.strip() != ""]

        # load fitur segmen-level
        rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', video_name + '.npy'))   # (32, 1024)
        flow_npy = np.load(os.path.join(self.path, 'all_flows', video_name + '.npy')) # (32, 1024)
        concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)                      # (32, 2048)

        # konversi frame-level interval → segmen-level label
        labels = self.frame_intervals_to_segment_labels(num_frames, gts, rgb_npy.shape[0])

        if self.modality == 'RGB':
            return torch.from_numpy(rgb_npy).float(), torch.from_numpy(labels).float()
        elif self.modality == 'FLOW':
            return torch.from_numpy(flow_npy).float(), torch.from_numpy(labels).float()
        else:
            return torch.from_numpy(concat_npy).float(), torch.from_numpy(labels).float()

    @staticmethod
    def frame_intervals_to_segment_labels(num_frames, intervals, num_segments=32):
        """
        Ubah anotasi frame-level → segmen-level label
        intervals contoh: [50,120, 200,250]
        """
        seg_length = num_frames / num_segments
        labels = np.zeros(num_segments, dtype=np.float32)

        for i in range(0, len(intervals), 2):
            start_f = intervals[i]
            end_f = intervals[i+1]
            start_seg = int(start_f // seg_length)
            end_seg   = int(end_f // seg_length)
            labels[start_seg:end_seg+1] = 1.0

        return labels


if __name__ == '__main__':
    # --- Load dataset ---
    loader_norm = Normal_Loader(is_train=1, modality='TWO')
    loader_anom = Anomaly_Loader(is_train=1, modality='TWO')

    # Test sample
    feats, lbls = loader_norm[0]
    print("Normal sample:", feats.shape, lbls.shape, lbls.sum())

    feats, lbls = loader_anom[0]
    print("Anomaly sample:", feats.shape, lbls.shape, "sum labels:", lbls.sum())

    # --- (1) Class imbalance check ---
    total_pos, total_neg = 0, 0
    for feats, lbls in loader_anom:
        lbls = lbls.numpy()
        total_pos += (lbls == 1).sum()
        total_neg += (lbls == 0).sum()
    for feats, lbls in loader_norm:
        lbls = lbls.numpy()
        total_pos += (lbls == 1).sum()
        total_neg += (lbls == 0).sum()

    print("\n[Class imbalance check]")
    print("Total segmen anomali (label=1):", total_pos)
    print("Total segmen normal  (label=0):", total_neg)
    print("Rasio positif:", total_pos / (total_pos + total_neg + 1e-8))

    # --- (2) Debug segmen mapping ---
    print("\n[Debug mapping sample anomaly videos]")
    for i in range(min(3, len(loader_anom))):  # cek 3 sample anomaly pertama
        feats, lbls = loader_anom[i]
        print(f"Video {i}: shape={feats.shape}, "
              f"jumlah label positif={lbls.sum()}, "
              f"distribusi label={np.unique(lbls.numpy(), return_counts=True)}")

    # --- (3) Histogram distribusi segmen positif per video ---
    pos_counts = []
    for feats, lbls in loader_anom:
        pos_counts.append(lbls.sum())

    plt.hist(pos_counts, bins=20)
    plt.xlabel("Jumlah segmen positif per video")
    plt.ylabel("Jumlah video")
    plt.title("Distribusi panjang anomali")
    plt.show()
