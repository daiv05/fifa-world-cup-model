"""
Dashboard Streamlit — El Oráculo del Balón (Mundial 2026).
Ejecutar con: streamlit run repository/src/visualization/dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

st.set_page_config(
    page_title="El Oráculo del Balón — Mundial 2026",
    page_icon="⚽",
    layout="wide",
)

st.title("El Oráculo del Balón — Mundial 2026")
st.caption("Simulación Monte Carlo con Machine Learning · 10,000 iteraciones")

page = st.sidebar.radio(
    "Sección",
    ["Probabilidades de Campeonato", "Comparación de Modelos", "Análisis de Features"],
)


@st.cache_data
def load_results() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "simulation_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_features() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "features.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


if page == "Probabilidades de Campeonato":
    df = load_results()
    if df is None:
        st.warning("No se encontraron resultados. Ejecuta primero `simulate.py --iterations 10000`.")
    else:
        top_n = st.slider("Mostrar Top N equipos", min_value=5, max_value=48, value=15)
        df_top = df.head(top_n)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_top["champion_pct"],
            y=df_top["team"],
            orientation="h",
            error_x=dict(
                type="data",
                symmetric=False,
                array=df_top["champion_ci_high"] - df_top["champion_pct"],
                arrayminus=df_top["champion_pct"] - df_top["champion_ci_low"],
            ),
            marker_color=px.colors.sequential.Blues_r[:top_n] if top_n <= 9 else None,
        ))
        fig.update_layout(
            title=f"Top {top_n} candidatos al campeonato",
            xaxis_title="Probabilidad de ser campeón (%)",
            yaxis=dict(autorange="reversed"),
            height=max(400, top_n * 28),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabla completa")
        st.dataframe(df, use_container_width=True)


elif page == "Comparación de Modelos":
    eval_path = PROCESSED_DIR / "model_evaluation.csv"
    if not eval_path.exists():
        st.warning("No se encontró `model_evaluation.csv`. Ejecuta primero `evaluate.py`.")
    else:
        eval_df = pd.read_csv(eval_path)
        st.subheader("Métricas de evaluación")
        st.dataframe(eval_df, use_container_width=True)

        fig = px.bar(
            eval_df,
            x="model",
            y=["log_loss", "brier_score"],
            barmode="group",
            title="Log-Loss y Brier Score por modelo (menor es mejor)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "**Log-Loss:** penaliza predicciones confiadas que resultan incorrectas. "
            "**Brier Score:** error cuadrático medio entre probabilidad predicha y resultado real."
        )


elif page == "Análisis de Features":
    df = load_features()
    if df is None:
        st.warning("No se encontraron features. Ejecuta primero `features.py`.")
    else:
        FEATURE_COLS = [
            "elo_diff", "squad_value_diff", "xg_avg_for",
            "xg_avg_against", "travel_distance_home", "travel_distance_away",
        ]
        available = [c for c in FEATURE_COLS if c in df.columns]

        st.subheader("Distribución de features por resultado")
        target_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        df["Resultado"] = df["target"].map(target_map)

        feat = st.selectbox("Feature", available)
        fig = px.histogram(df, x=feat, color="Resultado", barmode="overlay", nbins=50, opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlación entre features")
        corr = df[available].corr()
        fig2 = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f")
        st.plotly_chart(fig2, use_container_width=True)

        shap_img = Path("shap_summary.png")
        if shap_img.exists():
            st.subheader("SHAP — Importancia de variables (XGBoost)")
            st.image(str(shap_img))
