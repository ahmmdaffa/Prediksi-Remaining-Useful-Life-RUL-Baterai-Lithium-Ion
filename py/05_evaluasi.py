"""
Tahap 5 - Evaluasi

Script ini menghitung MAE, RMSE, dan R-squared, lalu membuat visualisasi
nilai aktual vs prediksi RUL.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUTPUT_DIR = Path("outputs_knn_rul")
MODEL_PATH = OUTPUT_DIR / "04_model_knn_gridsearch.pkl"
X_TEST_PATH = OUTPUT_DIR / "04_X_test.csv"
Y_TEST_PATH = OUTPUT_DIR / "04_y_test.csv"
TARGET_COLUMN = "RUL_Cycles"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model belum ditemukan. Jalankan 04_modelling_knn.py terlebih dahulu.")

    sns.set_theme(style="whitegrid", context="notebook")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET_COLUMN]

    # Melakukan prediksi pada data uji.
    y_pred = model.predict(X_test)

    # Menghitung metrik evaluasi regresi.
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics_df = pd.DataFrame(
        [
            {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Best_Params": str(model.best_params_),
            }
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / "05_metrics_evaluasi_knn.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "Actual_RUL": y_test.to_numpy(),
            "Predicted_RUL": y_pred,
            "Residual": y_test.to_numpy() - y_pred,
        }
    )
    prediction_df.to_csv(OUTPUT_DIR / "05_aktual_vs_prediksi_knn.csv", index=False)

    print("=" * 80)
    print("TAHAP 5 - EVALUASI MODEL")
    print("=" * 80)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("\nParameter terbaik:")
    print(model.best_params_)

    # Plot nilai aktual vs prediksi.
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.75)
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    plt.title("Aktual vs Prediksi RUL")
    plt.xlabel("RUL Aktual")
    plt.ylabel("RUL Prediksi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_plot_actual_vs_predicted_rul.png", dpi=300)
    plt.close()

    # Plot residual untuk melihat pola kesalahan prediksi.
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=y_pred, y=prediction_df["Residual"], alpha=0.75)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Plot Model KNN")
    plt.xlabel("Prediksi RUL")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_plot_residual_knn.png", dpi=300)
    plt.close()

    with open(OUTPUT_DIR / "05_analisis_evaluasi.txt", "w", encoding="utf-8") as file:
        file.write("TAHAP 5 - ANALISIS EVALUASI MODEL\n")
        file.write("=" * 80 + "\n\n")
        file.write(
            "Evaluasi model dilakukan menggunakan tiga metrik regresi, yaitu Mean "
            "Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-squared. "
            f"Hasil evaluasi menunjukkan MAE sebesar {mae:.4f}, RMSE sebesar "
            f"{rmse:.4f}, dan R-squared sebesar {r2:.4f}. MAE menunjukkan rata-rata "
            "kesalahan absolut prediksi RUL, sedangkan RMSE memberikan penalti lebih "
            "besar terhadap kesalahan yang ekstrem. R-squared menunjukkan proporsi "
            "variasi target RUL yang dapat dijelaskan oleh model.\n\n"
        )
        file.write(
            "Plot aktual vs prediksi digunakan untuk menilai kedekatan hasil prediksi "
            "dengan nilai sebenarnya. Apabila titik-titik data berada dekat dengan "
            "garis diagonal, maka model memiliki kemampuan prediksi yang baik. Plot "
            "residual digunakan untuk memeriksa pola kesalahan. Residual yang tersebar "
            "secara acak di sekitar nol menunjukkan bahwa model tidak meninggalkan "
            "pola kesalahan sistematis yang kuat.\n"
        )

    print("\nHasil evaluasi, grafik, dan analisis disimpan di folder outputs_knn_rul.")


if __name__ == "__main__":
    main()
