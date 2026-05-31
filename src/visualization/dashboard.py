import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.features.features import FEATURE_COLS

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
REPORTS_DIR = Path(__file__).parents[2] / "reports" / "figures"

st.set_page_config(
    page_title="Predicciones - Mundial 2026",
    page_icon=":crown:",
    layout="wide",
)

st.title("Predicciones - Mundial 2026")
st.caption("Simulación Monte Carlo con Machine Learning")

page = st.sidebar.radio(
    "Sección",
    [
        "Probabilidades de Campeonato",
        "Avance por Fase",
        "Comparación de Modelos",
        "Análisis de Features",
        "Sensibilidad a Lesiones",
    ],
)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


if page == "Probabilidades de Campeonato":
    df = load_csv("simulation_results.csv")
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
        ))
        fig.update_layout(
            title=f"Top {top_n} candidatos al campeonato (IC 95% Clopper-Pearson)",
            xaxis_title="Probabilidad de ser campeón (%)",
            yaxis=dict(autorange="reversed"),
            height=max(400, top_n * 28),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabla completa")
        st.dataframe(df, use_container_width=True)


elif page == "Avance por Fase":
    df = load_csv("tournament_progression.csv")
    if df is None:
        st.warning("No se encontró `tournament_progression.csv`. Ejecuta `simulate.py` con el modelo entrenado.")
    else:
        PHASES = ["group_stage", "round_of_32", "round_of_16",
                  "quarterfinals", "semifinals", "final", "champion"]
        PHASE_LABELS = {
            "group_stage": "Grupos",
            "round_of_32": "R32",
            "round_of_16": "R16",
            "quarterfinals": "Cuartos",
            "semifinals": "Semis",
            "final": "Final",
            "champion": "Campeón",
        }

        top_n = st.slider("Top N equipos por P(campeón)", 5, 48, 15)
        sub = df.head(top_n).copy()

        pct_cols = [f"{p}_pct" for p in PHASES]
        heat = sub.set_index("team")[pct_cols].copy()
        heat.columns = [PHASE_LABELS[p] for p in PHASES]

        fig = px.imshow(
            heat,
            text_auto=".1f",
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(x="Fase", y="Equipo", color="P (%)"),
            title="Probabilidad de llegar a cada fase",
        )
        fig.update_layout(height=max(400, top_n * 28))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabla completa de progresión")
        display_cols = ["team"] + [c for p in PHASES for c in (f"{p}_pct", f"{p}_ci_low", f"{p}_ci_high")]
        st.dataframe(df[display_cols], use_container_width=True)


elif page == "Comparación de Modelos":
    eval_df = load_csv("model_evaluation.csv")
    if eval_df is None:
        st.warning("No se encontró `model_evaluation.csv`. Ejecuta primero `evaluate.py`.")
    else:
        st.subheader("Métricas de evaluación (test temporal, date ≥ 2022)")
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
    df = load_csv("features.csv")
    if df is None:
        st.warning("No se encontraron features. Ejecuta primero `features.py`.")
    else:
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

        shap_img = REPORTS_DIR / "shap_summary.png"
        if shap_img.exists():
            st.subheader("SHAP - Importancia agregada de variables (XGBoost)")
            st.image(str(shap_img))


elif page == "Sensibilidad a Lesiones":
    df = load_csv("sensitivity_injuries.csv")
    if df is None:
        st.warning("No se encontró `sensitivity_injuries.csv`. Ejecuta `python -m src.analysis.sensitivity`.")
    else:
        st.subheader("Impacto de -30% squad_value sobre P(campeón)")
        st.dataframe(df, use_container_width=True)
        if "delta_champion_pct" in df.columns:
            fig = px.bar(
                df.sort_values("delta_champion_pct"),
                x="delta_champion_pct",
                y="team",
                orientation="h",
                title="Δ P(campeón) con lesión simulada",
            )
            st.plotly_chart(fig, use_container_width=True)
