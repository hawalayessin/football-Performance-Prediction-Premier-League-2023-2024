"""Current-season estimation, not a forecast of future player performance."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Never include outcomes or per-90 versions of those outcomes in the predictors.
NUMERIC_FEATURES = ["Age", "MP", "Starts", "Min", "90s", "xG", "xAG", "npxG", "PrgC", "PrgP", "xG_90", "xAG_90"]

def prepare_features(df):
    prepared = df.copy()
    columns = [name for name in NUMERIC_FEATURES if name in prepared.columns]
    for name in columns:
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    prepared["Pos"] = prepared.get("Pos", pd.Series("unknown", index=prepared.index)).fillna("unknown").astype(str)
    return prepared, columns + ["Pos"]

def split_players(df):
    groups = df["Player"].fillna("unknown").astype(str)
    if groups.nunique() < 5:
        raise ValueError("At least five distinct players are required for evaluation.")
    return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=groups))

def train_model(df, features, target):
    if target not in ("Gls", "Ast"):
        raise ValueError("Target must be Gls or Ast.")
    data, allowed = prepare_features(df)
    # Rebuild the allowlist here as well so callers cannot inject leaked features.
    columns = [name for name in allowed if name in features]
    if "Pos" not in columns:
        columns.append("Pos")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target]).reset_index(drop=True)
    train, test = split_players(data)
    numeric = [name for name in columns if name != "Pos"]
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Pos"]),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)),
    ])
    X, y = data[columns], data[target]
    # Fit imputation and encoding on training players only.
    model.fit(X.iloc[train], y.iloc[train])
    predictions = model.predict(X.iloc[test])
    baseline = DummyRegressor(strategy="mean").fit(X.iloc[train], y.iloc[train])
    metrics = {
        "MAE Train": mean_absolute_error(y.iloc[train], model.predict(X.iloc[train])),
        "MAE Test": mean_absolute_error(y.iloc[test], predictions),
        "R2 Test": r2_score(y.iloc[test], predictions),
        "Baseline MAE Test": mean_absolute_error(y.iloc[test], baseline.predict(X.iloc[test])),
        "Train players": data.iloc[train]["Player"].nunique(),
        "Test players": data.iloc[test]["Player"].nunique(),
    }
    return model, metrics
