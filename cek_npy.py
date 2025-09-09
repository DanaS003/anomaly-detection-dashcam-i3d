# cek dua file .npy apakah konsisten (shape, dtype, mean, std, dst)
import numpy as np

# NPY_NORMAL = "./UCF-Crime/all_flows/Abuse/Abuse001_x264.mp4.npy"
NPY_ANOM   = "./UCF-Crime/all_rgbs/Normal_Videos_event/Normal_Videos_003_x264.mp4.npy"
NPY_NORMAL   = "DashCam_00980_rgb (2).npy" 
# NPY_NORMAL   = "./DashCam/all_rgbs/Normal/DashCam_00006.mp4.npy"
def inspect_npy(path):
    a = np.load(path)
    print(f"=== {path} ===")
    print("shape :", a.shape)
    print("dtype :", a.dtype)
    print("min   :", a.min())
    print("max   :", a.max())
    print("mean  :", a.mean())
    print("std   :", a.std())
    print("contains NaN :", np.isnan(a).any())
    print("contains Inf :", np.isinf(a).any())
    print("first 10 vals:", a.flatten()[:10])
    print()
    return a

n_normal = inspect_npy(NPY_NORMAL)
n_anom   = inspect_npy(NPY_ANOM)

# perbandingan langsung
print("=== COMPARISON ===")
print("Same shape? ", n_normal.shape == n_anom.shape)
print("Same dtype? ", n_normal.dtype == n_anom.dtype)
print("Mean Normal vs Anomaly:", n_normal.mean(), n_anom.mean())
print("Std  Normal vs Anomaly:", n_normal.std(), n_anom.std())
