# Prediksi Remaining Useful Life (RUL) Baterai Lithium-Ion dengan KNN Regressor

Proyek ini membangun pipeline Data Mining end-to-end untuk memprediksi **Remaining Useful Life (RUL)** baterai Lithium-Ion menggunakan algoritma **K-Nearest Neighbors (KNN) Regressor**.

Proyek ini disusun untuk kebutuhan ujian proyek mata kuliah Proyek Data Mining. Dataset utama yang digunakan adalah `baterai.csv`, berisi data sensor baterai seperti nomor siklus, waktu pengukuran, tegangan, arus, suhu, dan kapasitas baterai.

## Ringkasan Proyek

| Komponen | Keterangan |
| --- | --- |
| Dataset | `baterai.csv` |
| Jumlah data mentah | 830.026 baris |
| Jumlah kolom | 7 kolom |
| Fitur utama | `Cycle_Number`, `Time_Seconds`, `Voltage_V`, `Current_A`, `Temperature_C`, `Capacity_Ah` |
| Target model | `RUL_Cycles` |
| Model utama | KNeighborsRegressor |
| Preprocessing utama | Agregasi per siklus, imputasi median, StandardScaler |
| Strategi validasi | Train-test split 80:20 dan GridSearchCV |
| Aplikasi web | Streamlit dashboard |

## Tujuan

Tujuan proyek ini adalah memprediksi sisa umur pakai baterai dalam satuan siklus. Karena dataset tidak memiliki kolom `RUL` eksplisit, target `RUL_Cycles` dibentuk dari data kapasitas baterai.

Definisi target:

```text
RUL_Cycles = EOL_Cycle - Cycle_Number
```

End-of-Life (EOL) didefinisikan ketika kapasitas baterai mencapai **80% dari kapasitas awal** masing-masing baterai. Jika baterai belum menyentuh ambang tersebut pada data yang tersedia, siklus terakhir digunakan sebagai pendekatan konservatif.

## Dataset

File dataset yang digunakan:

```text
baterai.csv
```

Kolom pada dataset:

| Kolom | Keterangan |
| --- | --- |
| `Battery_ID` | Identitas baterai |
| `Cycle_Number` | Nomor siklus charging/discharging |
| `Time_Seconds` | Waktu pengukuran dalam detik |
| `Voltage_V` | Tegangan baterai |
| `Current_A` | Arus baterai |
| `Temperature_C` | Suhu baterai |
| `Capacity_Ah` | Kapasitas baterai dalam Ah |

Hasil inspeksi awal menunjukkan bahwa dataset memiliki 830.026 baris, 7 kolom, dan tidak memiliki missing values pada data mentah.

## Alur Penelitian

Tahapan penelitian disusun sebagai berikut:

```text
1. Pengambilan Data & Inspeksi Awal
2. Exploratory Data Analysis (EDA)
3. Preprocessing
4. Modelling KNN Regressor
5. Evaluasi Model
6. Deployment Dashboard Streamlit
```

Setiap tahap tersedia dalam notebook terpisah agar alur pengerjaan mudah dipahami dan mudah dilampirkan ke laporan.

## Notebook Analisis

| Notebook | Isi |
| --- | --- |
| `01_pengambilan_data_inspeksi_awal.ipynb` | Load dataset, inspeksi kolom, tipe data, missing values, statistik deskriptif |
| `02_exploratory_data_analysis.ipynb` | Visualisasi kapasitas vs siklus, SOH vs siklus, heatmap korelasi |
| `03_preprocessing.ipynb` | Agregasi data level siklus, pembentukan target RUL, scaling preview |
| `04_modelling_knn.ipynb` | Train-test split, pipeline KNN, GridSearchCV |
| `05_evaluasi.ipynb` | MAE, RMSE, R-squared, plot aktual vs prediksi, residual plot |

## Preprocessing

Data mentah berisi banyak baris pengukuran untuk satu siklus baterai. Agar model mempelajari pola degradasi per siklus, data diagregasi menjadi data level siklus.

Fitur hasil agregasi meliputi:

```text
Time_Seconds_max
Voltage_V_mean, Voltage_V_std, Voltage_V_min, Voltage_V_max
Current_A_mean, Current_A_std, Current_A_min, Current_A_max
Temperature_C_mean, Temperature_C_std, Temperature_C_min, Temperature_C_max
Capacity_Ah
SOH_Percent
Normalized_Cycle
```

Preprocessing utama:

- Missing values ditangani menggunakan imputasi median.
- Fitur numerik dipisahkan dari target `RUL_Cycles`.
- Feature scaling menggunakan `StandardScaler`.
- Scaling diterapkan di dalam pipeline modelling untuk mengurangi risiko data leakage.

## Modelling

Model utama yang digunakan adalah:

```text
KNeighborsRegressor
```

Pipeline model:

```text
SimpleImputer(strategy="median")
StandardScaler()
KNeighborsRegressor()
```

GridSearchCV digunakan untuk mencari kombinasi parameter terbaik:

```text
n_neighbors: 3, 5, 7, 9, 11, 15, 21, 31
weights: uniform, distance
metric: euclidean, manhattan
```

Parameter terbaik yang diperoleh:

```text
metric: manhattan
n_neighbors: 3
weights: distance
```

## Hasil Evaluasi

Evaluasi dilakukan menggunakan data uji sebesar 20% dari data level siklus.

| Metrik | Nilai |
| --- | ---: |
| MAE | 6.1106 |
| RMSE | 24.5644 |
| R-squared | 0.9570 |

Interpretasi singkat:

- **MAE** menunjukkan rata-rata kesalahan absolut prediksi RUL.
- **RMSE** memberi penalti lebih besar pada kesalahan prediksi yang ekstrem.
- **R-squared** sebesar 0.9570 menunjukkan bahwa model mampu menjelaskan sebagian besar variasi target RUL pada data uji.

## Aplikasi Web

Proyek ini juga memiliki dashboard web berbasis Streamlit:

```text
app.py
```

Dashboard berisi:

- Ringkasan dataset dan model
- Inspeksi data awal
- EDA dengan visualisasi kapasitas, SOH, dan korelasi
- Insight EDA otomatis, seperti degradasi terbesar dan korelasi terkuat terhadap RUL
- Preprocessing level siklus
- Modelling KNN dan hasil GridSearchCV
- Evaluasi model
- Plot aktual vs prediksi
- Residual plot
- Filter Battery ID dengan mode semua baterai atau pilihan manual
- Upload CSV opsional

## Struktur Folder

```text
UTS_ProyekDataMining/
|-- app.py
|-- baterai.csv
|-- requirements.txt
|-- README.md
|-- 01_pengambilan_data_inspeksi_awal.ipynb
|-- 02_exploratory_data_analysis.ipynb
|-- 03_preprocessing.ipynb
|-- 04_modelling_knn.ipynb
|-- 05_evaluasi.ipynb
|-- py/
|   |-- 01_pengambilan_data_inspeksi_awal.py
|   |-- 02_exploratory_data_analysis.py
|   |-- 03_preprocessing.py
|   |-- 04_modelling_knn.py
|   `-- 05_evaluasi.py
`-- outputs_knn_rul/
    |-- file hasil analisis, model, metrik, dan visualisasi eksperimen
```

Catatan: folder `outputs_knn_rul` berisi output eksperimen dari versi script Python. Untuk penggunaan dashboard Streamlit, aplikasi tidak bergantung pada folder tersebut karena proses analisis dihitung langsung dari `baterai.csv`.

## Cara Menjalankan Proyek

### 1. Install dependency

```bash
pip install -r requirements.txt
```

### 2. Menjalankan dashboard Streamlit

```bash
python -m streamlit run app.py --server.address localhost --server.port 8501
```

Setelah server berjalan, buka:

```text
http://localhost:8501
```

### 3. Menjalankan notebook

Notebook dapat dibuka langsung melalui VS Code atau Jupyter Notebook. Jalankan berurutan:

```text
01_pengambilan_data_inspeksi_awal.ipynb
02_exploratory_data_analysis.ipynb
03_preprocessing.ipynb
04_modelling_knn.ipynb
05_evaluasi.ipynb
```

## Catatan Implementasi

- Target `RUL_Cycles` dibentuk karena dataset tidak memiliki kolom RUL eksplisit.
- Batas EOL default adalah 80% dari kapasitas awal baterai.
- KNN dipilih karena sederhana, intuitif, dan cocok sebagai baseline regresi berbasis kedekatan pola data.
- Feature scaling wajib digunakan karena KNN sensitif terhadap jarak.
- `GridSearchCV` pada dashboard memakai `n_jobs=1` agar stabil saat dijalankan di laptop atau VS Code.
- Dashboard Streamlit dibuat dengan custom CSS agar tampilannya lebih rapi dan tidak terlalu standar.

## Kesimpulan

Berdasarkan pipeline yang dibangun, KNN Regressor dapat digunakan sebagai model baseline untuk memprediksi Remaining Useful Life baterai Lithium-Ion. Hasil evaluasi menunjukkan performa yang baik dengan nilai R-squared sebesar 0.9570. Model masih dapat dikembangkan lebih lanjut dengan membandingkan algoritma lain seperti Random Forest Regressor, Gradient Boosting, atau Support Vector Regression.
