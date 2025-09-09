import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx].strip()+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx].strip()+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        else:
            # --- Normal Loader (test) ---
            if isinstance(self, Normal_Loader):
                parts = self.data_list[idx].strip().split()
                name, frames, gts = parts[0], int(parts[1]), int(parts[2])
            # --- Anomaly Loader (test) ---
            else:
                parts = self.data_list[idx].strip().split('|')
                name, frames, gts_str = parts[0], int(parts[1]), parts[2].strip()
                gts = gts_str.strip("[]").split(',')
                gts = [int(i) for i in gts if i.strip() != ""]

            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)

            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            else:
                return concat_npy, gts, frames


class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-1].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            else:
                return concat_npy, gts, frames

if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))
    #print(loader[1], loader2[1])
