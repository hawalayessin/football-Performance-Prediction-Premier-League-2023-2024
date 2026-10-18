# Football Analytics and Same-Season Estimation

Academic Python project combining a Streamlit dashboard, Random Forest regression
and optional Gemini commentary for Premier League 2023/2024 player statistics.

## What the model estimates

The model estimates the observed goals or assists of held-out players using other
statistics from the **same season**. It does not predict next season or future matches.
Inputs include playing time, age, position and expected-goal statistics. Outcomes
and their per-90 derivatives (`Gls_90`, `Ast_90`) are excluded from predictors.

Players are grouped for an 80/20 split so a transferred player cannot appear in
both training and evaluation sets. Imputation and encoding are fitted only on
training players. The evaluation screen reports held-out MAE, R² and mean-baseline
MAE; it does not claim a measured score until training has actually run.

## Run locally

Use Python 3.11+:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
streamlit run app.py
```

Copy `.env.example` to `.env` only if enabling Gemini commentary. Supply your own
key and a model available to your account. The application works without Gemini.
Published keys from earlier revisions must be revoked; deleting them from the
current code does not invalidate them or erase earlier commits.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover feature leakage, player overlap, training-only preprocessing and
unseen positions. GitHub Actions runs them for pushes and pull requests.

## Data and reproducibility

`premier-player-23-24.csv` is the dataset already included in the academic project.
Its original provider and redistribution license are not documented in this repo;
confirm both before redistributing it elsewhere. The CSV is read as observed
historical data. A future-performance study would need multiple seasons and a
temporal split using information available before the predicted period.

The app trains its own pipeline and can export it. Older root `.joblib` artifacts
are legacy outputs, not the pipeline evaluated here. It does not accept uploaded
pickled models, because loading an untrusted pickle/joblib can execute code.

Author: [Yassine Ben Hawala](https://portfoliohawala.vercel.app/).
