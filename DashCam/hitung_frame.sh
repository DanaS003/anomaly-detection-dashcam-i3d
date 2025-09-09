#!/bin/bash
# Script untuk rename file di folder ANOMALY/NORMAL dengan pilihan user

# Fungsi untuk menghapus .mp4
hapus_mp4() {
    echo "Anda memilih Opsi 1: Menghapus '.mp4' dari nama file."
    folders=(
        "all_rgbs/ANOMALY"
        "all_rgbs/NORMAL"
        "all_flows/ANOMALY"
        "all_flows/NORMAL"
    )

    for folder in "${folders[@]}"; do
        echo "Processing folder: $folder"
        find "$folder" -type f -name "*.mp4.npy" | while read -r file; do
            dir=$(dirname "$file")
            base=$(basename "$file")
            newname="${base/.mp4.npy/.npy}"
            mv "$file" "$dir/$newname"
            echo "Renamed: $file -> $dir/$newname"
        done
    done
    echo "Selesai menghapus '.mp4' dari semua file!"
}

# Fungsi untuk menambahkan .mp4
tambah_mp4() {
    echo "Anda memilih Opsi 2: Menambahkan '.mp4' ke nama file."
    folders=(
        "all_rgbs/ANOMALY"
        "all_rgbs/NORMAL"
        "all_flows/ANOMALY"
        "all_flows/NORMAL"
    )

    for folder in "${folders[@]}"; do
        echo "Processing folder: $folder"
        find "$folder" -type f -name "*.npy" | while read -r file; do
            if [[ "$file" != *".mp4.npy" ]]; then
                dir=$(dirname "$file")
                base=$(basename "$file")
                newname="${base/.npy/.mp4.npy}"
                mv "$file" "$dir/$newname"
                echo "Renamed: $file -> $dir/$newname"
            fi
        done
    done
    echo "Selesai menambahkan '.mp4' ke semua file!"
}

# Menu pilihan untuk user
echo "Pilih opsi untuk rename file:"
echo "1. Hapus '.mp4' dari nama file (misal: file.mp4.npy -> file.npy)"
echo "2. Tambahkan '.mp4' ke nama file (misal: file.npy -> file.mp4.npy)"
echo -n "Masukkan nomor opsi Anda (1 atau 2): "
read option

# Logika untuk menjalankan fungsi berdasarkan input user
case $option in
    1)
        hapus_mp4
        ;;
    2)
        tambah_mp4
        ;;
    *)
        echo "Pilihan tidak valid. Silakan jalankan skrip lagi dan pilih 1 atau 2."
        exit 1
        ;;
esac

echo "Skrip selesai dijalankan."