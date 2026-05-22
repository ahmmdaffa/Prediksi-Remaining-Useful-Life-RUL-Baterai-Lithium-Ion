from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("baterai.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20


st.set_page_config(
    page_title="KNN RUL Baterai",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #171916;
        --muted: #667066;
        --line: #dfe5dc;
        --paper: #fbfcf8;
        --wash: #f3f6ef;
        --teal: #0f8f7d;
        --coral: #d96f4d;
        --gold: #b48a22;
        --leaf: #5f8d3d;
    }

    .stApp {
        background: var(--wash);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #161914;
        border-right: 1px solid #2c3228;
    }

    [data-testid="stSidebar"] * {
        color: #f4f7ee;
    }

    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        padding: 0.35rem 0.2rem;
        border-radius: 0;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }

    .top-band {
        border-top: 5px solid var(--teal);
        border-bottom: 1px solid var(--line);
        background: var(--paper);
        padding: 1.4rem 1.5rem 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 0.35rem 0;
    }

    h1 {
        color: var(--ink);
        font-size: 2.2rem !important;
        line-height: 1.12 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.35rem !important;
    }

    h2, h3 {
        color: var(--ink);
        letter-spacing: 0 !important;
    }

    .lead {
        color: var(--muted);
        max-width: 860px;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0;
    }

    .metric-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        padding: 1rem 1.05rem;
        min-height: 105px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 850;
        line-height: 1.15;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .analysis-block {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 5px solid var(--gold);
        padding: 1rem 1.1rem;
        margin-top: 0.6rem;
        color: #343a32;
        line-height: 1.55;
    }

    .analysis-block strong {
        color: var(--ink);
    }

    .subtle-divider {
        height: 1px;
        background: var(--line);
        margin: 1.2rem 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
sns.set_theme(style="whitegrid", context="notebook")


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="top-band">
            <p class="eyebrow">Lithium-Ion Battery RUL</p>
            <h1>{title}</h1>
            <p class="lead">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analysis_block(text: str) -> None:
    st.markdown(f'<div class="analysis-block">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_default_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if not DATA_PATH.exists():
        st.error(f"File dataset tidak ditemukan: {DATA_PATH.resolve()}")
        st.stop()
    return load_default_data(str(DATA_PATH))


@st.cache_data(show_spinner=False)
def build_cycle_level_dataset(data: pd.DataFrame, eol_ratio: float) -> pd.DataFrame:
    data = data.sort_values(["Battery_ID", "Cycle_Number", "Time_Seconds"]).copy()

    cycle_df = (
        data.groupby(["Battery_ID", "Cycle_Number"], as_index=False)
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

    std_columns = [col for col in cycle_df.columns if col.endswith("_std")]
    cycle_df[std_columns] = cycle_df[std_columns].fillna(0)

    cycle_df["Initial_Capacity_Ah"] = cycle_df.groupby("Battery_ID")["Capacity_Ah"].transform("first")
    cycle_df["SOH_Percent"] = cycle_df["Capacity_Ah"] / cycle_df["Initial_Capacity_Ah"] * 100

    processed_parts = []
    for _, part in cycle_df.groupby("Battery_ID", sort=False):
        part = part.sort_values("Cycle_Number").copy()
        initial_capacity = part["Initial_Capacity_Ah"].iloc[0]
        eol_capacity = eol_ratio * initial_capacity

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


@st.cache_data(show_spinner=False)
def prepare_features(cycle_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_column = "RUL_Cycles"
    drop_columns = ["Battery_ID", "EOL_Cycle", target_column]
    X = cycle_df.drop(columns=drop_columns).select_dtypes(include="number")
    y = cycle_df[target_column]
    return X, y


@st.cache_resource(show_spinner=False)
def train_knn_model(X: pd.DataFrame, y: pd.Series) -> tuple[GridSearchCV, pd.DataFrame, pd.Series, np.ndarray, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor()),
        ]
    )

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "CV_RMSE": -grid_search.best_score_,
        "Train_Size": len(X_train),
        "Test_Size": len(X_test),
    }
    return grid_search, X_test, y_test, y_pred, metrics


def make_capacity_plot(plot_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="Cycle_Number",
        y="Capacity_Ah",
        hue="Battery_ID",
        marker="o",
        linewidth=1.8,
        ax=ax,
    )
    ax.set_title("Tren Penurunan Kapasitas terhadap Siklus", fontsize=13, weight="bold")
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Capacity (Ah)")
    ax.legend(title="Battery ID", frameon=False)
    fig.tight_layout()
    return fig


def make_soh_plot(plot_df: pd.DataFrame, eol_ratio: float):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="Cycle_Number",
        y="SOH_Percent",
        hue="Battery_ID",
        marker="o",
        linewidth=1.8,
        ax=ax,
    )
    ax.axhline(eol_ratio * 100, color="#d96f4d", linestyle="--", linewidth=1.6, label="Batas EOL")
    ax.set_title("State of Health terhadap Siklus", fontsize=13, weight="bold")
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("SOH (%)")
    ax.legend(title="Battery ID", frameon=False)
    fig.tight_layout()
    return fig


def make_corr_plot(cycle_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = [
        "Cycle_Number",
        "Time_Seconds_max",
        "Voltage_V_mean",
        "Current_A_mean",
        "Temperature_C_mean",
        "Capacity_Ah",
        "SOH_Percent",
        "RUL_Cycles",
    ]
    corr = cycle_df[cols].corr()
    sns.heatmap(corr, cmap="Spectral_r", center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title("Korelasi Fitur Numerik Utama", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def make_actual_pred_plot(y_test: pd.Series, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.72, color="#0f8f7d", edgecolor="#ffffff", linewidth=0.4, ax=ax)
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    ax.plot([min_value, max_value], [min_value, max_value], color="#d96f4d", linestyle="--", linewidth=1.8)
    ax.set_title("Aktual vs Prediksi RUL", fontsize=13, weight="bold")
    ax.set_xlabel("RUL Aktual")
    ax.set_ylabel("RUL Prediksi")
    fig.tight_layout()
    return fig


def make_residual_plot(y_pred: np.ndarray, residual: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=y_pred, y=residual, alpha=0.72, color="#5f8d3d", edgecolor="#ffffff", linewidth=0.4, ax=ax)
    ax.axhline(0, color="#d96f4d", linestyle="--", linewidth=1.8)
    ax.set_title("Residual Plot Model KNN", fontsize=13, weight="bold")
    ax.set_xlabel("Prediksi RUL")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    return fig


with st.sidebar:
    st.markdown("### KNN RUL Baterai")
    st.caption("Dashboard proyek Data Mining")
    uploaded_file = st.file_uploader("Dataset CSV", type=["csv"])
    eol_ratio = st.slider("Batas End-of-Life", min_value=0.70, max_value=0.90, value=0.80, step=0.01)
    page = st.radio(
        "Tahap",
        [
            "Dashboard",
            "Data & Inspeksi",
            "EDA",
            "Preprocessing",
            "Modelling KNN",
            "Evaluasi",
        ],
    )


df = load_data(uploaded_file)
required_columns = {
    "Battery_ID",
    "Cycle_Number",
    "Time_Seconds",
    "Voltage_V",
    "Current_A",
    "Temperature_C",
    "Capacity_Ah",
}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    st.error(f"Kolom wajib tidak ditemukan: {missing_columns}")
    st.stop()

cycle_df = build_cycle_level_dataset(df, eol_ratio)
X, y = prepare_features(cycle_df)
battery_options = sorted(cycle_df["Battery_ID"].unique())

with st.sidebar:
    selected_batteries = st.multiselect(
        "Battery ID",
        options=battery_options,
        default=battery_options,
    )

plot_df = cycle_df[cycle_df["Battery_ID"].isin(selected_batteries)]
if plot_df.empty:
    st.warning("Pilih minimal satu Battery ID untuk visualisasi.")
    st.stop()


if page == "Dashboard":
    render_header(
        "Dashboard Prediksi RUL Baterai",
        "Ringkasan pipeline prediksi Remaining Useful Life baterai lithium-ion menggunakan KNN Regressor.",
    )

    with st.spinner("Melatih model KNN dan menghitung metrik evaluasi..."):
        model, X_test, y_test, y_pred, metrics = train_knn_model(X, y)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Data Mentah", f"{len(df):,}", "baris sensor", "#0f8f7d")
    with c2:
        metric_card("Data Siklus", f"{len(cycle_df):,}", "baris hasil agregasi", "#5f8d3d")
    with c3:
        metric_card("RMSE", f"{metrics['RMSE']:.2f}", "kesalahan regresi", "#d96f4d")
    with c4:
        metric_card("R-squared", f"{metrics['R2']:.3f}", "variasi target terjelaskan", "#b48a22")

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.18, 1])
    with left:
        st.pyplot(make_capacity_plot(plot_df), clear_figure=True)
    with right:
        st.pyplot(make_actual_pred_plot(y_test, y_pred), clear_figure=True)

    analysis_block(
        "<strong>Inti hasil:</strong> model KNN mampu mempelajari pola degradasi baterai dari fitur level siklus. "
        "Dashboard ini menampilkan hubungan kapasitas, SOH, target RUL, parameter terbaik, dan evaluasi prediksi "
        "tanpa perlu membuka file output tambahan."
    )


elif page == "Data & Inspeksi":
    render_header(
        "Pengambilan Data & Inspeksi Awal",
        "Tahap awal untuk memastikan dataset terbaca, kolom sesuai, dan kualitas data cukup layak untuk dianalisis.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Baris", f"{df.shape[0]:,}", "observasi sensor", "#0f8f7d")
    with c2:
        metric_card("Kolom", f"{df.shape[1]}", "atribut dataset", "#5f8d3d")
    with c3:
        metric_card("Baterai", f"{df['Battery_ID'].nunique()}", "unit unik", "#d96f4d")
    with c4:
        metric_card("Missing", f"{int(df.isna().sum().sum()):,}", "nilai hilang", "#b48a22")

    st.subheader("Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Missing Values")
        st.dataframe(df.isna().sum().to_frame("Jumlah Missing Value"), use_container_width=True)
    with right:
        st.subheader("Tipe Data")
        dtype_df = pd.DataFrame({"Kolom": df.dtypes.index, "Tipe Data": df.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    analysis_block(
        "<strong>Analisis:</strong> dataset berisi data sensor baterai lithium-ion seperti siklus, waktu, tegangan, "
        "arus, suhu, dan kapasitas. Inspeksi awal memastikan tidak ada masalah struktur data sebelum masuk ke EDA, "
        "preprocessing, dan pemodelan."
    )


elif page == "EDA":
    render_header(
        "Exploratory Data Analysis",
        "Visualisasi degradasi kapasitas, State of Health, dan korelasi fitur untuk memahami perilaku baterai.",
    )

    tab_capacity, tab_soh, tab_corr = st.tabs(["Kapasitas", "SOH", "Korelasi"])
    with tab_capacity:
        st.pyplot(make_capacity_plot(plot_df), clear_figure=True)
    with tab_soh:
        st.pyplot(make_soh_plot(plot_df, eol_ratio), clear_figure=True)
    with tab_corr:
        st.pyplot(make_corr_plot(cycle_df), clear_figure=True)

    analysis_block(
        "<strong>Analisis:</strong> tren kapasitas terhadap siklus menunjukkan pola degradasi baterai. "
        "SOH membantu menempatkan kapasitas terhadap kapasitas awal, sedangkan heatmap korelasi memberi gambaran "
        "fitur mana yang berhubungan dengan kapasitas dan RUL."
    )


elif page == "Preprocessing":
    render_header(
        "Preprocessing",
        "Data sensor mentah diagregasi ke level siklus, target RUL dibentuk, lalu fitur disiapkan untuk model KNN.",
    )

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled_preview = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Fitur", f"{X.shape[1]}", "numerik", "#0f8f7d")
    with c2:
        metric_card("Target", "RUL", "dalam siklus", "#5f8d3d")
    with c3:
        metric_card("EOL", f"{eol_ratio:.0%}", "dari kapasitas awal", "#d96f4d")
    with c4:
        metric_card("Missing Fitur", f"{int(X.isna().sum().sum()):,}", "setelah agregasi", "#b48a22")

    st.subheader("Dataset Level Siklus")
    st.dataframe(cycle_df.head(25), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Fitur Sebelum Scaling")
        st.dataframe(X.head(10), use_container_width=True)
    with right:
        st.subheader("Preview Setelah StandardScaler")
        st.dataframe(X_scaled_preview.head(10), use_container_width=True)

    analysis_block(
        "<strong>Analisis:</strong> satu siklus memiliki banyak pengukuran sensor, sehingga data diagregasi menggunakan "
        "rata-rata, standar deviasi, nilai minimum, dan maksimum. Target <strong>RUL_Cycles</strong> dibentuk dari sisa "
        "siklus menuju End-of-Life. Scaling wajib untuk KNN karena algoritma ini sensitif terhadap skala fitur."
    )


elif page == "Modelling KNN":
    render_header(
        "Modelling KNN Regressor",
        "Model dibangun dengan pipeline imputasi, scaling, KNN Regressor, dan GridSearchCV untuk mencari parameter terbaik.",
    )

    with st.spinner("Melatih model KNN dengan GridSearchCV..."):
        model, X_test, y_test, y_pred, metrics = train_knn_model(X, y)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Train", f"{metrics['Train_Size']:,}", "baris", "#0f8f7d")
    with c2:
        metric_card("Test", f"{metrics['Test_Size']:,}", "baris", "#5f8d3d")
    with c3:
        metric_card("CV RMSE", f"{metrics['CV_RMSE']:.2f}", "validasi silang", "#d96f4d")
    with c4:
        metric_card("Best K", str(model.best_params_["knn__n_neighbors"]), "tetangga", "#b48a22")

    st.subheader("Parameter Terbaik")
    best_params_df = pd.DataFrame([model.best_params_]).T.rename(columns={0: "Nilai"})
    st.dataframe(best_params_df, use_container_width=True)

    st.subheader("Grid Search")
    cv_results = pd.DataFrame(model.cv_results_)
    cv_results = cv_results[
        [
            "param_knn__n_neighbors",
            "param_knn__weights",
            "param_knn__metric",
            "mean_test_score",
            "rank_test_score",
        ]
    ].copy()
    cv_results["RMSE_CV"] = -cv_results["mean_test_score"]
    cv_results = cv_results.sort_values("rank_test_score")
    st.dataframe(cv_results.head(12), use_container_width=True)

    analysis_block(
        "<strong>Analisis:</strong> KNN Regressor memprediksi RUL berdasarkan kedekatan pola fitur dengan data historis. "
        "GridSearchCV mengevaluasi beberapa kombinasi jumlah tetangga, bobot, dan metrik jarak. Pipeline menjaga proses "
        "imputasi dan scaling tetap berada di dalam validasi sehingga mengurangi risiko data leakage."
    )


elif page == "Evaluasi":
    render_header(
        "Evaluasi Model",
        "Performa model dinilai menggunakan MAE, RMSE, R-squared, plot aktual vs prediksi, dan residual plot.",
    )

    with st.spinner("Menghitung evaluasi model..."):
        model, X_test, y_test, y_pred, metrics = train_knn_model(X, y)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("MAE", f"{metrics['MAE']:.2f}", "rata-rata error absolut", "#0f8f7d")
    with c2:
        metric_card("RMSE", f"{metrics['RMSE']:.2f}", "error dengan penalti besar", "#d96f4d")
    with c3:
        metric_card("R-squared", f"{metrics['R2']:.3f}", "variasi target terjelaskan", "#b48a22")

    residual = y_test.to_numpy() - y_pred
    left, right = st.columns([1, 1])
    with left:
        st.pyplot(make_actual_pred_plot(y_test, y_pred), clear_figure=True)
    with right:
        st.pyplot(make_residual_plot(y_pred, residual), clear_figure=True)

    st.subheader("Contoh Hasil Prediksi")
    prediction_df = pd.DataFrame(
        {
            "Actual_RUL": y_test.to_numpy(),
            "Predicted_RUL": y_pred,
            "Residual": residual,
        }
    )
    st.dataframe(prediction_df.head(30), use_container_width=True)

    analysis_block(
        f"<strong>Analisis:</strong> model menghasilkan MAE sebesar <strong>{metrics['MAE']:.4f}</strong>, "
        f"RMSE sebesar <strong>{metrics['RMSE']:.4f}</strong>, dan R-squared sebesar "
        f"<strong>{metrics['R2']:.4f}</strong>. Plot aktual vs prediksi menunjukkan kedekatan hasil prediksi terhadap "
        "nilai sebenarnya, sedangkan residual plot digunakan untuk memeriksa pola kesalahan model."
    )
