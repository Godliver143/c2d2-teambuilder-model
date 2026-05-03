"""
scripts/train_mission_model.py

Retrain the mission recommendation model from the latest evaluation data.
Run this whenever new evaluation rows are added to the database or CSV.

Usage:
    python scripts/train_mission_model.py
    python scripts/train_mission_model.py --csv path/to/evaluations.csv
    python scripts/train_mission_model.py --from-db   (uses DATABASE_URL env var)

Output: models/mission_rf_model.pkl + models/model_metadata.json
"""
import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SCORE_COLS = [
    "planning", "attention_to_detail", "time_management",
    "decisiveness", "tactics",
    "ump_planning", "ump_attention_to_detail", "ump_time_management",
    "ump_decisiveness", "ump_tactics",
]


def load_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_from_db() -> pd.DataFrame:
    import os
    from sqlalchemy import create_engine, text
    engine = create_engine(os.environ["DATABASE_URL"])
    query = """
        SELECT
            e.id, e.leader_id, l.rank AS leader_rank, l.name AS leader_name,
            l.unit AS leader_unit, ev.name AS event_name, ev.event_date,
            e.planning, e.attention_to_detail, e.time_management,
            e.decisiveness, e.tactics,
            e.ump_planning, e.ump_attention_to_detail, e.ump_time_management,
            e.ump_decisiveness, e.ump_tactics
        FROM evaluations e
        JOIN leaders l ON l.id = e.leader_id
        JOIN events ev ON ev.id = e.event_id
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def extract_mission_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mission_type"] = df["event_name"].str.extract(r"- (.+)$")[0].str.strip().str.upper()
    df = df.dropna(subset=["mission_type"])
    return df


def augment(X: np.ndarray, y: np.ndarray, n_synthetic: int = 700) -> tuple:
    """
    Gaussian noise augmentation that preserves real-data correlations.
    Oversamples low-performing rows (score < 3.5) at 4× to balance the dataset,
    which skews toward 4.0+ in real data.
    """
    np.random.seed(42)
    weights = np.where(y < 3.5, 4.0, np.where(y < 4.0, 2.0, 1.0))
    weights /= weights.sum()

    rows_x, rows_y = [], []
    for _ in range(n_synthetic):
        idx = np.random.choice(len(X), p=weights)
        new_x = np.clip(X[idx].astype(float) + np.random.normal(0, 0.25, X.shape[1]), 1, 5)
        # Mission encoded column stays integer — resample from real
        new_x[-1] = X[np.random.randint(len(X))][-1]
        rows_x.append(new_x)
        rows_y.append(float(np.mean(new_x[:-1])))  # composite of score cols

    return (
        np.vstack([X, np.array(rows_x)]),
        np.concatenate([y, np.array(rows_y)]),
    )


def train(df: pd.DataFrame) -> dict:
    df = extract_mission_type(df)
    log.info("Loaded %d rows, %d unique leaders, %d mission types",
             len(df), df["leader_id"].nunique(), df["mission_type"].nunique())

    mission_classes = sorted(df["mission_type"].unique().tolist())
    mission_map = {m: i for i, m in enumerate(mission_classes)}
    df["mission_encoded"] = df["mission_type"].map(mission_map)

    df["performance_score"] = df[SCORE_COLS].mean(axis=1)

    feature_cols = SCORE_COLS + ["mission_encoded"]
    X = df[feature_cols].values.astype(float)
    y = df["performance_score"].values

    X_aug, y_aug = augment(X, y)
    log.info("Augmented: %d → %d rows", len(X), len(X_aug))

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_aug, y_aug)

    # Validate on real data only
    mae_real = mean_absolute_error(y, model.predict(X))
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 5),
                                scoring="neg_mean_absolute_error")
    cv_mae = float(-cv_scores.mean())
    cv_std = float(cv_scores.std())

    log.info("MAE on real data:   %.3f", mae_real)
    log.info("CV MAE (5-fold):    %.3f ± %.3f", cv_mae, cv_std)

    # Feature importances
    log.info("\nFeature importances:")
    for col, imp in sorted(zip(feature_cols, model.feature_importances_),
                           key=lambda x: -x[1]):
        log.info("  %-35s %.3f", col, imp)

    # Save
    joblib.dump(model, MODEL_DIR / "mission_rf_model.pkl")

    metadata = {
        "feature_cols": feature_cols,
        "score_cols": SCORE_COLS,
        "mission_classes": mission_classes,
        "training_rows_real": int(len(X)),
        "training_rows_augmented": int(len(X_aug)),
        "rf_mae_real": round(mae_real, 4),
        "cv_mae": round(cv_mae, 4),
        "cv_std": round(cv_std, 4),
        "score_range": [1.0, 5.0],
        "target_mean": round(float(y.mean()), 3),
        "target_std": round(float(y.std()), 3),
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("\n✓ Model saved to models/mission_rf_model.pkl")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data_local/evaluations_full.csv",
                        help="Path to CSV file")
    parser.add_argument("--from-db", action="store_true",
                        help="Load from DATABASE_URL instead of CSV")
    args = parser.parse_args()

    if args.from_db:
        df = load_from_db()
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            csv_path = Path("evaluations_full.csv")
        df = load_from_csv(str(csv_path))

    train(df)
