"""
Train and evaluate the churn prediction model.

Artifacts written:
- model.pkl: trained Random Forest classifier (keeps predict.py compatible)
- columns.pkl: expected model input columns (keeps predict.py compatible)
- model_metrics.json: held-out evaluation metrics
- feature_importance.csv: global Random Forest feature importances

Run from the project directory:
    python train.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"


def find_target_column(frame: pd.DataFrame) -> str:
    """Find a likely target column, preferring a column named Churn."""
    exact = [col for col in frame.columns if str(col).strip().lower() == "churn"]
    if exact:
        return exact[0]

    candidates = [
        col for col in frame.columns
        if any(token in str(col).strip().lower() for token in ("churn", "exit", "target"))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Could not identify the target column. Expected a column named "
            "'Churn' or a column containing 'churn', 'exit', or 'target'."
        )
    raise ValueError(f"Multiple possible target columns found: {candidates}")


def encode_target(series: pd.Series) -> pd.Series:
    """Normalize common binary target representations to 0/1."""
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "churn": 1,
        "stay": 0,
        "stayed": 0,
        "left": 1,
        "retained": 0,
    }
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        unexpected = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError(
            "The target column must contain two recognizable classes. "
            f"Unrecognized values: {unexpected[:10]}"
        )
    if encoded.nunique() != 2:
        raise ValueError(
            "The target column must contain exactly two classes after encoding."
        )
    return encoded.astype(int)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATASET_PATH.name} in {BASE_DIR}. "
            "Run this script from the project folder or place dataset.csv there."
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        raise ValueError("dataset.csv contains no rows.")

    target_column = find_target_column(df)

    # Convert TotalCharges before missing-value handling, as it can contain blanks.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop only clear identifier columns, not every column containing the letters "id".
    identifier_columns = [
        col for col in df.columns
        if str(col).strip().lower() in {
            "id", "customerid", "customer_id", "customer id", "rowid", "row_id"
        }
    ]
    df = df.drop(columns=identifier_columns, errors="ignore")

    # Ensure target is still present after identifier cleanup.
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} was removed unexpectedly.")

    df = df.dropna().copy()
    if df.empty:
        raise ValueError("No rows remain after removing records with missing values.")

    y = encode_target(df[target_column])
    raw_X = df.drop(columns=[target_column])

    # One-hot encode categoricals. Keep the exact feature order for predict.py.
    X = pd.get_dummies(raw_X)
    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing.")

    # Stratification keeps the churn/stay ratio similar in train and test sets.
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except (IndexError, AttributeError):
            y_score = None

    metrics = {
        "model": "RandomForestClassifier",
        "target_column": target_column,
        "dataset_rows_after_cleaning": int(len(df)),
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "positive_class": "Churn",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix_labels": ["Stay (0)", "Churn (1)"],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Stay", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_score is not None and y_test.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
    else:
        metrics["roc_auc"] = None

    # Save held-out predictions so the dashboard can plot real evaluation curves.
    # These rows come only from the test split and are not training predictions.
    if y_score is not None:
        test_predictions = pd.DataFrame({
            "actual_churn": y_test.to_numpy(dtype=int),
            "churn_probability": y_score.astype(float),
        })
        threshold_rows = []
        for threshold in [round(value, 2) for value in __import__("numpy").arange(0.10, 0.91, 0.05)]:
            threshold_pred = (y_score >= threshold).astype(int)
            threshold_rows.append({
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_test, threshold_pred)),
                "precision": float(precision_score(y_test, threshold_pred, zero_division=0)),
                "recall": float(recall_score(y_test, threshold_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, threshold_pred, zero_division=0)),
                "true_positive": int(((y_test.to_numpy() == 1) & (threshold_pred == 1)).sum()),
                "false_positive": int(((y_test.to_numpy() == 0) & (threshold_pred == 1)).sum()),
                "true_negative": int(((y_test.to_numpy() == 0) & (threshold_pred == 0)).sum()),
                "false_negative": int(((y_test.to_numpy() == 1) & (threshold_pred == 0)).sum()),
            })
        threshold_metrics = pd.DataFrame(threshold_rows)
    else:
        test_predictions = pd.DataFrame()
        threshold_metrics = pd.DataFrame()

    model_path = BASE_DIR / "model.pkl"
    columns_path = BASE_DIR / "columns.pkl"
    metrics_path = BASE_DIR / "model_metrics.json"
    importance_path = BASE_DIR / "feature_importance.csv"
    predictions_path = BASE_DIR / "test_predictions.csv"
    threshold_metrics_path = BASE_DIR / "threshold_metrics.csv"

    # Preserve the artifact names expected by the existing predict.py.
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    with columns_path.open("wb") as file:
        pickle.dump(list(X.columns), file)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importance.to_csv(importance_path, index=False)
    if not test_predictions.empty:
        test_predictions.to_csv(predictions_path, index=False)
        threshold_metrics.to_csv(threshold_metrics_path, index=False)

    print("\n=== Churn model evaluation (held-out test set) ===")
    print(f"Dataset rows after cleaning: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1_score']:.3f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")

    print("\nConfusion matrix (rows=actual [Stay, Churn], columns=predicted [Stay, Churn]):")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Stay", "Actual Churn"],
        columns=["Predicted Stay", "Predicted Churn"],
    ))

    print("\nTop 10 global feature importances (not causal explanations):")
    print(importance.head(10).to_string(index=False))

    print("\nSaved artifacts:")
    print(f"- {model_path.name}")
    print(f"- {columns_path.name}")
    print(f"- {metrics_path.name}")
    print(f"- {importance_path.name}")
    if not test_predictions.empty:
        print(f"- {predictions_path.name}")
        print(f"- {threshold_metrics_path.name}")
    print("\nNote: metrics describe this held-out split; they are not a guarantee of real-world performance.")


if __name__ == "__main__":
    main()
