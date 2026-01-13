# ⚽ Football Performance Prediction + Gemini AI

## 📌 Description du projet
Cette application est un projet **BI & Intelligence Artificielle** qui combine :
- **Machine Learning (RandomForest)** pour prédire les performances des joueurs
- **IA générative (Google Gemini)** pour fournir une analyse experte et contextuelle
- **Streamlit** pour une interface web interactive

Le projet est basé sur les données de la **Premier League 2023/2024**.

---

## 🎯 Objectifs
- Analyser les performances individuelles des joueurs
- Prédire :
  - le nombre de buts (**Gls**)
  - le nombre de passes décisives (**Ast**)
- Générer une analyse qualitative via **Gemini AI**
- Fournir une application exploitable dans un contexte **BI / Data Science**

---

## 🧠 Technologies utilisées

| Catégorie | Technologies |
|---------|-------------|
| Frontend | Streamlit |
| Data | Pandas |
| Visualisation | Plotly |
| Machine Learning | Scikit-learn (RandomForest) |
| IA Générative | Google Gemini |
| Modèles | Joblib |
| Langage | Python 3.9+ |

---

## 📂 Structure du projet

```text
football-streamlit/
│── app.py
│── premier-player-23-24.csv
│── requirements.txt
│── README.md
streamlit run app.py
