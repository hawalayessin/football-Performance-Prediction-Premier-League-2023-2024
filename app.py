# =========================================================
# app.py – Streamlit | Football Prediction + Gemini AI
# VERSION PROD – STABLE
# =========================================================

import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import google.generativeai as genai

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Football Performance Prediction + Gemini",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Performance Prediction – Premier League 2023/2024")
st.markdown("""
Application **BI & IA** combinant :
- **Machine Learning (RandomForest)** pour la prédiction
- **Gemini AI** pour l’analyse experte
""")

# =========================================================
# CONFIG GEMINI (MODE PROD LOCAL)
# =========================================================
GEMINI_API_KEY = "AIzaSyDqlPkMxraqUhQOExngIQrIgkkIy5rNZqk"

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    gemini_model = None

# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("premier-player-23-24.csv")
    except Exception:
        return None

# =========================================================
# FEATURE ENGINEERING (DETERMINISTE)
# =========================================================
def prepare_features(df):
    df = df.copy()

    def simplify_position(pos):
        pos = str(pos).upper()
        if any(p in pos for p in ["FW", "ST", "LW", "RW", "CF"]):
            return "ATT"
        if any(p in pos for p in ["MF", "CM", "DM", "AM"]):
            return "MID"
        if any(p in pos for p in ["DF", "CB", "LB", "RB"]):
            return "DEF"
        if "GK" in pos:
            return "GK"
        return "OTHER"

    df["pos_simple"] = df["Pos"].apply(simplify_position)

    numeric_cols = [
        "Age", "MP", "Starts", "Min", "90s",
        "xG", "xAG", "npxG", "PrgC", "PrgP",
        "Gls_90", "Ast_90", "xG_90", "xAG_90"
    ]

    features = []
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            features.append(col)

    pos_dummies = pd.get_dummies(df["pos_simple"], prefix="pos")
    df = pd.concat([df, pos_dummies], axis=1)
    features.extend(pos_dummies.columns.tolist())

    return df, features

# =========================================================
# ENTRAÎNEMENT MODÈLE (FEATURES SAUVEGARDÉES)
# =========================================================
def train_model(df, features, target):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    metrics = {
        "MAE Train": mean_absolute_error(y_train, model.predict(X_train)),
        "MAE Test": mean_absolute_error(y_test, model.predict(X_test)),
        "R2 Train": r2_score(y_train, model.predict(X_train)),
        "R2 Test": r2_score(y_test, model.predict(X_test)),
    }

    return model, metrics

# =========================================================
# ANALYSE GEMINI
# =========================================================
def gemini_analysis(player):
    if not gemini_model:
        return "Analyse Gemini indisponible."

    prompt = f"""
Tu es un analyste football professionnel.

Nom : {player['Player']}
Équipe : {player['Team']}
Position : {player['Pos']}
Âge : {player['Age']}
Minutes jouées : {player['Min']}
xG : {player['xG']}
xAG : {player['xAG']}

Analyse sa performance offensive et son potentiel.
Réponse courte et structurée.
"""
    return gemini_model.generate_content(prompt).text

# =========================================================
# SIDEBAR
# =========================================================
section = st.sidebar.radio(
    "Navigation",
    ["Aperçu Données", "Entraînement ML", "Prédiction Joueur"]
)

# =========================================================
# APERÇU DONNÉES
# =========================================================
if section == "Aperçu Données":
    df = load_data()
    if df is None:
        st.error("Dataset introuvable")
    else:
        st.dataframe(df.head(), use_container_width=True)
        fig = px.histogram(df, x="Gls", title="Distribution des buts")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ENTRAÎNEMENT ML
# =========================================================
elif section == "Entraînement ML":
    df = load_data()
    if df is not None:
        target = st.selectbox("Variable cible", ["Gls", "Ast"])
        df_prep, features = prepare_features(df)

        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement en cours..."):
                model, metrics = train_model(df_prep, features, target)

            for k, v in metrics.items():
                st.metric(k, f"{v:.2f}")

            buffer = io.BytesIO()
            joblib.dump(
                {"model": model, "features": features},
                buffer
            )
            buffer.seek(0)

            st.download_button(
                "Télécharger le modèle",
                data=buffer,
                file_name=f"rf_{target}_prod.joblib",
                mime="application/octet-stream"
            )

# =========================================================
# PRÉDICTION (COMPATIBLE ANCIENS & NOUVEAUX MODÈLES)
# =========================================================
elif section == "Prédiction Joueur":
    df = load_data()
    if df is not None:
        target = st.selectbox("Variable cible", ["Gls", "Ast"])
        model_file = st.file_uploader("Charger modèle ML (.joblib)", type="joblib")

        if model_file:
            loaded = joblib.load(model_file)

            if isinstance(loaded, RandomForestRegressor):
                model = loaded
                trained_features = model.feature_names_in_
                st.warning("Ancien modèle détecté – compatibilité limitée")
            else:
                model = loaded["model"]
                trained_features = loaded["features"]

            df_prep, _ = prepare_features(df)

            player_name = st.selectbox(
                "Choisir joueur",
                df_prep["Player"].unique()
            )
            player = df_prep[df_prep["Player"] == player_name].iloc[0]

            X = pd.DataFrame([player])

            for col in trained_features:
                if col not in X.columns:
                    X[col] = 0

            X = X[trained_features]

            prediction = model.predict(X)[0]

            c1, c2 = st.columns(2)
            c1.metric("Prédiction ML", f"{prediction:.2f}")
            c2.metric("Valeur réelle", f"{player[target]}")

            if st.button("Analyse IA Gemini"):
                with st.spinner("Analyse IA..."):
                    analysis = gemini_analysis(player)
                st.subheader("Analyse Gemini")
                st.write(analysis)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center'>Projet BI & IA – Version PROD</div>",
    unsafe_allow_html=True
)
