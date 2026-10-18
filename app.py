"""Streamlit demonstration of same-season estimation with held-out players."""
import io
import os
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from modeling import prepare_features, train_model

load_dotenv()
st.set_page_config(page_title="Football analytics", page_icon="⚽", layout="wide")
st.title("Football analytics — Premier League 2023/2024")
st.caption("Academic demonstration: same-season estimation, not a forecast of future seasons.")

@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).with_name("premier-player-23-24.csv"))

@st.cache_resource
def fit_target(target):
    prepared, features = prepare_features(load_data())
    return train_model(prepared, features, target)

df = load_data()
section = st.sidebar.radio("Navigation", ["Dataset", "Evaluation", "Player estimation"])
if section == "Dataset":
    st.dataframe(df.head(), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Gls", title="Goals distribution"), use_container_width=True)
else:
    target = st.selectbox("Target", ["Gls", "Ast"])
    model, metrics = fit_target(target)
    if section == "Evaluation":
        st.write("Players are separated between training and test sets, including players appearing for multiple clubs.")
        st.json(metrics)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        st.download_button("Download locally trained model", buffer.getvalue(), file_name=f"football_{target}.joblib")
    else:
        name = st.selectbox("Player", sorted(df["Player"].dropna().unique()))
        player = df[df["Player"] == name].iloc[0]
        prepared, _ = prepare_features(pd.DataFrame([player]))
        st.metric("Model estimate", f"{model.predict(prepared)[0]:.2f}")
        st.metric("Observed same-season total", f"{player[target]}")
        st.info("This demonstration may display a training player. Use the held-out evaluation above to assess accuracy.")
        if st.button("Generate Gemini commentary"):
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                st.warning("Set GEMINI_API_KEY locally to enable this optional feature.")
            else:
                from google import genai
                try:
                    with genai.Client(api_key=key) as client:
                        result = client.models.generate_content(
                            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                            contents=f"Summarize these observed football statistics, without claiming a future forecast: player={player['Player']}, team={player['Team']}, minutes={player['Min']}, goals={player['Gls']}, assists={player['Ast']}.")
                    st.write(result.text)
                except Exception:
                    st.error("Commentary unavailable. Check the local key, model availability and quota.")
