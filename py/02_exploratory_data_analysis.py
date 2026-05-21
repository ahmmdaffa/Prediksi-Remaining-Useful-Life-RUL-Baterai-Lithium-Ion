"""
Tahap 2 - Exploratory Data Analysis (EDA)

Script ini membuat visualisasi untuk melihat pola degradasi kapasitas baterai
terhadap siklus dan korelasi antar fitur sensor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("baterai.csv")
OUTPUT_DIR = Path("outputs_knn_rul")


def build_cycle_level_data(df: pd.DataFrame) -> pd.DataFrame:
    # Mengagregasi data sensor mentah menjadi data level siklus.
    # Setiap baris hasil agregasi merepresentasikan satu Battery_ID pada satu Cycle_Number.
    cycle_df = (
        df.groupby(["Battery_ID", "Cycle_Number"], as_index=False)
        .agg(
            Time_Seconds_max=("Time_Seconds", "max"),
            Voltage_V_mean=("Voltage_V", "mean"),
            Current_A_mean=("Current_A", "mean"),
            Temperature_C_mean=("Temperature_C", "mean"),
            Capacity_Ah=("Capacity_Ah", "mean"),
        )
        .sort_values(["Battery_ID", "Cycle_Number"])
    )

    # Membentuk indikator State of Health (SOH) untuk membantu analisis degradasi.
    cycle_df["Initial_Capacity_Ah"] = cycle_df.groupby("Battery_ID")["Capacity_Ah"].transform("first")
    cycle_df["SOH_Percent"] = cycle_df["Capacity_Ah"] / cycle_df["Initial_Capacity_Ah"] * 100
    return cycle_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    df = pd.read_csv(DATA_PATH)
    cycle_df = build_cycle_level_data(df)

    print("=" * 80)
    print("TAHAP 2 - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    print(f"Ukuran data level siklus untuk EDA: {cycle_df.shape[0]:,} baris")
    print("\nContoh data level siklus:")
    print(cycle_df.head())

    # Visualisasi tren penurunan kapasitas terhadap siklus.
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=cycle_df,
        x="Cycle_Number",
        y="Capacity_Ah",
        hue="Battery_ID",
        marker="o",
        linewidth=1.6,
    )
    plt.title("Tren Penurunan Kapasitas Baterai terhadap Siklus")
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacity (Ah)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_eda_capacity_vs_cycle.png", dpi=300)
    plt.close()

    # Visualisasi tren SOH terhadap siklus.
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=cycle_df,
        x="Cycle_Number",
        y="SOH_Percent",
        hue="Battery_ID",
        marker="o",
        linewidth=1.6,
    )
    plt.axhline(80, color="red", linestyle="--", label="Batas EOL 80%")
    plt.title("Tren State of Health (SOH) Baterai terhadap Siklus")
    plt.xlabel("Cycle Number")
    plt.ylabel("SOH (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_eda_soh_vs_cycle.png", dpi=300)
    plt.close()

    # Heatmap korelasi antar fitur numerik.
    plt.figure(figsize=(10, 8))
    corr = cycle_df.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Heatmap Korelasi Fitur Numerik")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_eda_correlation_heatmap.png", dpi=300)
    plt.close()

    # Menyimpan draft analisis EDA.
    with open(OUTPUT_DIR / "02_analisis_eda.txt", "w", encoding="utf-8") as file:
        file.write("TAHAP 2 - ANALISIS EXPLORATORY DATA ANALYSIS (EDA)\n")
        file.write("=" * 80 + "\n\n")
        file.write(
            "EDA dilakukan untuk memahami karakteristik awal data baterai sebelum "
            "model dibangun. Data sensor mentah diagregasi ke level siklus agar "
            "tren degradasi dapat diamati dengan lebih jelas. Visualisasi kapasitas "
            "terhadap Cycle_Number digunakan untuk melihat apakah kapasitas baterai "
            "menurun seiring bertambahnya siklus pemakaian. Pola penurunan tersebut "
            "merupakan ciri umum degradasi baterai lithium-ion akibat proses "
            "charging-discharging berulang.\n\n"
        )
        file.write(
            "Selain itu, visualisasi SOH menunjukkan persentase kesehatan baterai "
            "dibandingkan kapasitas awal. Garis 80% digunakan sebagai batas umum "
            "End-of-Life (EOL). Heatmap korelasi digunakan untuk memeriksa hubungan "
            "antara fitur sensor, kapasitas, dan siklus. Korelasi ini membantu "
            "mengidentifikasi fitur yang berpotensi relevan terhadap prediksi "
            "Remaining Useful Life (RUL).\n"
        )

    print("\nGrafik dan analisis EDA disimpan di folder outputs_knn_rul.")


if __name__ == "__main__":
    main()
