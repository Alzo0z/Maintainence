"""Train and save the three classifiers from the report so the Streamlit app
can load them at startup instead of retraining each time.

Run once before launching the app:

    python train_models.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Rotational speed [rpm]", "Torque [Nm]"]
TARGET = "Machine failure"
RANDOM_STATE = 42


def build_pipeline(model):
    pre = ColumnTransformer([("num", StandardScaler(), FEATURES)])
    return Pipeline([("preprocessor", pre), ("model", model)])


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(50, 25), activation="relu", solver="adam",
            max_iter=500, random_state=RANDOM_STATE,
        ),
    }

    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        # Score for confirmation
        acc = pipe.score(X_test, y_test)
        path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(pipe, path)
        print(f"  trained {name:22s} -> {path.name}  (test acc {acc:.3f})")

    # Save train/test split as well so the boundary plot uses identical data
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df_train.to_parquet(MODELS_DIR / "train.parquet")
    df_test.to_parquet(MODELS_DIR / "test.parquet")
    print(f"  saved train.parquet ({len(df_train)} rows) "
          f"and test.parquet ({len(df_test)} rows)")


if __name__ == "__main__":
    main()
