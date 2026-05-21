"""
Tahap 1 - Pengambilan Data & Inspeksi Awal

Script ini membaca dataset baterai.csv, menampilkan struktur awal data,
dan menyimpan ringkasan inspeksi ke folder outputs_knn_rul.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("baterai.csv")
OUTPUT_DIR = Path("outputs_knn_rul")


def main() -> None:
    # Membuat folder output untuk menyimpan hasil inspeksi.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Membaca dataset menggunakan pandas.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    # Menampilkan informasi dasar dataset ke terminal.
    print("=" * 80)
    print("TAHAP 1 - PENGAMBILAN DATA & INSPEKSI AWAL")
    print("=" * 80)
    print(f"Ukuran dataset: {df.shape[0]:,} baris dan {df.shape[1]:,} kolom")
    print("\nDaftar kolom:")
    print(df.columns.tolist())
    print("\nLima baris pertama:")
    print(df.head())
    print("\nInformasi tipe data:")
    df.info()
    print("\nJumlah missing values per kolom:")
    print(df.isna().sum())
    print("\nStatistik deskriptif fitur numerik:")
    print(df.describe())

    # Menyimpan ringkasan inspeksi agar mudah dilampirkan ke laporan.
    with open(OUTPUT_DIR / "01_ringkasan_inspeksi_awal.txt", "w", encoding="utf-8") as file:
        file.write("TAHAP 1 - PENGAMBILAN DATA & INSPEKSI AWAL\n")
        file.write("=" * 80 + "\n\n")
        file.write(f"Ukuran dataset: {df.shape[0]:,} baris dan {df.shape[1]:,} kolom\n\n")
        file.write("Daftar kolom:\n")
        file.write(str(df.columns.tolist()) + "\n\n")
        file.write("Lima baris pertama:\n")
        file.write(df.head().to_string() + "\n\n")
        file.write("Jumlah missing values per kolom:\n")
        file.write(df.isna().sum().to_string() + "\n\n")
        file.write("Statistik deskriptif fitur numerik:\n")
        file.write(df.describe().to_string() + "\n\n")
        file.write("Analisis Hasil:\n")
        file.write(
            "Dataset berhasil dimuat menggunakan pandas. Dataset berisi data sensor "
            "baterai lithium-ion, yaitu Battery_ID, Cycle_Number, Time_Seconds, "
            "Voltage_V, Current_A, Temperature_C, dan Capacity_Ah. Inspeksi awal "
            "dilakukan untuk memahami ukuran data, tipe data, nilai hilang, serta "
            "rentang nilai numerik. Tahap ini penting karena kualitas data awal "
            "akan memengaruhi validitas proses EDA, preprocessing, pemodelan, "
            "dan evaluasi model prediksi Remaining Useful Life (RUL).\n"
        )

    print(f"\nRingkasan inspeksi disimpan di: {OUTPUT_DIR / '01_ringkasan_inspeksi_awal.txt'}")


if __name__ == "__main__":
    main()
