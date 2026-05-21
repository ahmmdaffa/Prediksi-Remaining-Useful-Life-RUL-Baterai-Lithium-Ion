"""
Tahap 3 - Preprocessing

Script ini menangani missing values, membentuk target RUL_Cycles,
memisahkan fitur dan target, serta menyiapkan dataset untuk modelling.
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("baterai.csv")
OUTPUT_DIR = Path("outputs_knn_rul")
EOL_THRESHOLD_RATIO = 0.80


def build_cycle_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Mengurutkan data agar agregasi dan perhitungan RUL konsisten.
    df = df.sort_values(["Battery_ID", "Cycle_Number", "Time_Seconds"]).copy()

    # Mengagregasi data sensor per siklus dengan statistik ringkas.
    cycle_df = (
        df.groupby(["Battery_ID", "Cycle_Number"], as_index=False)
        .agg(
            Time_Seconds_max=("Time_Seconds", "max"),
            Voltage_V_mean=("Voltage_V", "mean"),
            Voltage_V_std=("Voltage_V", "std"),
            Voltage_V_min=("Voltage_V", "min"),
            Voltage_V_max=("Voltage_V", "max"),
            Current_A_mean=("Current_A", "mean"),
            Current_A_std=("Current_A", "std"),
            Current_A_min=("Current_A", "min"),
            Current_A_max=("Current_A", "max"),
            Temperature_C_mean=("Temperature_C", "mean"),
            Temperature_C_std=("Temperature_C", "std"),
            Temperature_C_min=("Temperature_C", "min"),
            Temperature_C_max=("Temperature_C", "max"),
            Capacity_Ah=("Capacity_Ah", "mean"),
        )
        .sort_values(["Battery_ID", "Cycle_Number"])
    )

    # Nilai standar deviasi dapat NaN jika suatu siklus hanya memiliki satu observasi.
    std_columns = [col for col in cycle_df.columns if col.endswith("_std")]
    cycle_df[std_columns] = cycle_df[std_columns].fillna(0)

    # Membentuk SOH sebagai indikator kesehatan baterai.
    cycle_df["Initial_Capacity_Ah"] = cycle_df.groupby("Battery_ID")["Capacity_Ah"].transform("first")
    cycle_df["SOH_Percent"] = cycle_df["Capacity_Ah"] / cycle_df["Initial_Capacity_Ah"] * 100

    # Membentuk target RUL_Cycles berdasarkan sisa siklus menuju End-of-Life.
    processed_parts = []
    for battery_id, part in cycle_df.groupby("Battery_ID", sort=False):
        part = part.sort_values("Cycle_Number").copy()
        initial_capacity = part["Initial_Capacity_Ah"].iloc[0]
        eol_capacity = EOL_THRESHOLD_RATIO * initial_capacity

        below_eol = part.loc[part["Capacity_Ah"] <= eol_capacity, "Cycle_Number"]
        eol_cycle = below_eol.iloc[0] if not below_eol.empty else part["Cycle_Number"].max()

        part["EOL_Cycle"] = eol_cycle
        part["RUL_Cycles"] = (eol_cycle - part["Cycle_Number"]).clip(lower=0)
        part["Normalized_Cycle"] = (
            (part["Cycle_Number"] - part["Cycle_Number"].min())
            / max(part["Cycle_Number"].max() - part["Cycle_Number"].min(), 1)
        )
        processed_parts.append(part)

    return pd.concat(processed_parts, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    cycle_df = build_cycle_level_dataset(df)

    # Target utama proyek adalah RUL_Cycles.
    target_column = "RUL_Cycles"
    drop_columns = ["Battery_ID", "EOL_Cycle", target_column]
    X = cycle_df.drop(columns=drop_columns).select_dtypes(include="number")
    y = cycle_df[target_column]

    # Imputasi median digunakan sebagai perlindungan jika masih ada missing values.
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scaling dilakukan karena KNN sensitif terhadap skala fitur.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Dataset modelling menyimpan fitur asli numerik dan target.
    # Scaling final tetap dilakukan lagi dalam pipeline modelling untuk mencegah data leakage.
    modeling_df = X.copy()
    modeling_df[target_column] = y.values
    scaled_preview_df = X_scaled.copy()
    scaled_preview_df[target_column] = y.values

    cycle_df.to_csv(OUTPUT_DIR / "03_cycle_level_dataset.csv", index=False)
    modeling_df.to_csv(OUTPUT_DIR / "03_modeling_dataset_unscaled.csv", index=False)
    scaled_preview_df.to_csv(OUTPUT_DIR / "03_modeling_dataset_scaled_preview.csv", index=False)

    print("=" * 80)
    print("TAHAP 3 - PREPROCESSING")
    print("=" * 80)
    print(f"Data level siklus: {cycle_df.shape[0]:,} baris dan {cycle_df.shape[1]:,} kolom")
    print(f"Jumlah fitur numerik: {X.shape[1]}")
    print(f"Target: {target_column}")
    print("\nMissing values setelah agregasi:")
    print(modeling_df.isna().sum())
    print("\nContoh data siap modelling:")
    print(modeling_df.head())

    with open(OUTPUT_DIR / "03_analisis_preprocessing.txt", "w", encoding="utf-8") as file:
        file.write("TAHAP 3 - ANALISIS PREPROCESSING\n")
        file.write("=" * 80 + "\n\n")
        file.write(
            "Preprocessing dilakukan untuk mengubah data sensor mentah menjadi data "
            "yang sesuai untuk pemodelan regresi. Karena satu siklus memiliki banyak "
            "baris pengukuran, data diagregasi ke level siklus menggunakan statistik "
            "rata-rata, standar deviasi, nilai minimum, dan nilai maksimum. Strategi "
            "ini membuat setiap baris merepresentasikan kondisi baterai pada satu "
            "siklus tertentu.\n\n"
        )
        file.write(
            f"Dataset tidak memiliki kolom RUL eksplisit, sehingga target RUL_Cycles "
            f"dibentuk berdasarkan sisa siklus menuju End-of-Life. End-of-Life "
            f"didefinisikan ketika kapasitas baterai mencapai {EOL_THRESHOLD_RATIO:.0%} "
            f"dari kapasitas awal masing-masing baterai. Missing values ditangani "
            f"dengan imputasi median. Feature scaling menggunakan StandardScaler "
            f"dilakukan karena KNN menghitung kedekatan data berdasarkan jarak, "
            f"sehingga fitur dengan skala besar dapat mendominasi hasil prediksi "
            f"jika tidak distandarkan.\n"
        )

    print("\nHasil preprocessing disimpan di folder outputs_knn_rul.")


if __name__ == "__main__":
    main()
