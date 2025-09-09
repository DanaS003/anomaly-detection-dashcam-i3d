#!/bin/bash

# Folder dasar
FLOWS_DIR="all_flows"
RGBS_DIR="all_rgbs"

# Pilihan menu
echo "Pilih opsi:"
echo "1. Sinkronisasi ANOMALY"
echo "2. Sinkronisasi NORMAL"
read -p "Masukkan pilihan (1/2): " choice

if [ "$choice" == "1" ]; then
    SUBDIR="ANOMALY"
    TXT_FILE="train_anomaly.txt"
elif [ "$choice" == "2" ]; then
    SUBDIR="NORMAL"
    TXT_FILE="train_normal.txt"
else
    echo "Pilihan tidak valid!"
    exit 1
fi

FLOW_PATH="$FLOWS_DIR/$SUBDIR"
RGB_PATH="$RGBS_DIR/$SUBDIR"

echo "[INFO] Mengecek folder: $SUBDIR"
echo "[INFO] Flow path: $FLOW_PATH"
echo "[INFO] RGB path : $RGB_PATH"

# Pastikan folder ada
if [ ! -d "$FLOW_PATH" ] || [ ! -d "$RGB_PATH" ]; then
    echo "[ERROR] Salah satu folder tidak ditemukan!"
    exit 1
fi

# Ambil daftar nama file tanpa ekstensi
flow_files=$(ls "$FLOW_PATH"/*.npy 2>/dev/null | xargs -n1 basename | sed 's/\.npy$//')
rgb_files=$(ls "$RGB_PATH"/*.npy 2>/dev/null | xargs -n1 basename | sed 's/\.npy$//')

# Hapus file di RGBS yang tidak ada di FLOWS
for f in $rgb_files; do
    if ! echo "$flow_files" | grep -q "^$f$"; then
        echo "[DELETE] $RGB_PATH/$f.npy (tidak ada di $FLOW_PATH)"
        rm -f "$RGB_PATH/$f.npy"
    fi
done

# Update daftar file setelah sinkron
final_files=$(ls "$FLOW_PATH"/*.npy 2>/dev/null | xargs -n1 basename | sed 's/\.npy$//' | sort)

# Update file .txt
> "$TXT_FILE"  # kosongkan dulu
for f in $final_files; do
    echo "$f.mp4" >> "$TXT_FILE"
done

echo "[DONE] Sinkronisasi selesai!"
echo "[INFO] File list disimpan ke $TXT_FILE"
