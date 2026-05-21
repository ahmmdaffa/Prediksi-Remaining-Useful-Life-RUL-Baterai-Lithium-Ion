"""
Tahap 4 - Modelling KNN

Script ini melakukan train-test split, membangun KNeighborsRegressor,
dan menggunakan GridSearchCV untuk mencari parameter terbaik.
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs_knn_rul")
MODELING_DATA_PATH = OUTPUT_DIR / "03_modeling_dataset_unscaled.csv"
TARGET_COLUMN = "RUL_Cycles"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            "File hasil preprocessing belum ditemukan. Jalankan 03_preprocessing.py terlebih dahulu."
        )

    df = pd.read_csv(MODELING_DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Membagi data menjadi data latih dan data uji.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Pipeline digunakan agar imputasi dan scaling dipelajari hanya dari data latih.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor()),
        ]
    )

    candidate_neighbors = [3, 5, 7, 9, 11, 15, 21, 31]
    valid_neighbors = [k for k in candidate_neighbors if k < len(X_train)]
    if not valid_neighbors:
        valid_neighbors = [1]

    param_grid = {
        "knn__n_neighbors": valid_neighbors,
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=min(5, len(X_train)),
        n_jobs=-1,
        verbose=1,
    )

    print("=" * 80)
    print("TAHAP 4 - MODELLING KNN REGRESSOR")
    print("=" * 80)
    print(f"Jumlah data latih: {len(X_train):,}")
    print(f"Jumlah data uji: {len(X_test):,}")
    print("\nProses GridSearchCV dimulai...")

    grid_search.fit(X_train, y_train)

    # Menyimpan artefak model dan data uji untuk tahap evaluasi.
    with open(OUTPUT_DIR / "04_model_knn_gridsearch.pkl", "wb") as file:
        pickle.dump(grid_search, file)

    X_test.to_csv(OUTPUT_DIR / "04_X_test.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "04_y_test.csv", index=False)

    best_rmse = -grid_search.best_score_

    print("\nParameter terbaik:")
    print(grid_search.best_params_)
    print(f"Skor CV terbaik (RMSE): {best_rmse:.4f}")

    with open(OUTPUT_DIR / "04_analisis_modelling.txt", "w", encoding="utf-8") as file:
        file.write("TAHAP 4 - ANALISIS MODELLING KNN\n")
        file.write("=" * 80 + "\n\n")
        file.write(
            "Model yang digunakan adalah K-Nearest Neighbors Regressor. KNN merupakan "
            "algoritma berbasis instance yang memprediksi nilai kontinu berdasarkan "
            "kedekatan suatu data terhadap sejumlah tetangga terdekatnya. Pada "
            "penelitian ini, data dibagi menjadi data latih dan data uji dengan "
            f"rasio {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}. Pipeline "
            "terdiri dari imputasi median, StandardScaler, dan KNN Regressor.\n\n"
        )
        file.write(
            "GridSearchCV digunakan untuk mencari kombinasi parameter terbaik, "
            "meliputi jumlah tetangga, jenis pembobotan, dan metrik jarak. "
            f"Parameter terbaik yang diperoleh adalah {grid_search.best_params_}. "
            f"Skor cross-validation terbaik berdasarkan RMSE adalah {best_rmse:.4f}. "
            "Penggunaan pipeline membantu mencegah data leakage karena scaler hanya "
            "dipelajari dari data latih pada setiap proses validasi.\n"
        )

    print("\nModel dan hasil modelling disimpan di folder outputs_knn_rul.")


if __name__ == "__main__":
    main()
