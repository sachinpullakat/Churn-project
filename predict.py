"""Prediction helpers for the ChurnIQ customer churn project.

The model probability is a model score, not a guarantee or necessarily a
calibrated real-world probability. Threshold selection should be validated
on held-out data before operational use.
"""
from pathlib import Path
import pickle

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"


def _load_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact '{path.name}' was not found in {BASE_DIR}. "
            "Run train.py first to create the model files."
        )
    with path.open("rb") as file:
        return pickle.load(file)


# Keep the project's original artifact names for compatibility.
model = _load_artifact(MODEL_PATH)
columns = _load_artifact(COLUMNS_PATH)


def preprocess_input(data):
    """One-hot encode a single customer and align columns with training data."""
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Discard unknown columns and preserve the exact training feature order.
    df = df[columns]
    return df


def _get_churn_probability(processed):
    """Return P(class=1), locating class 1 explicitly instead of assuming order."""
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The saved model does not support predict_proba(). "
            "Retrain with a classifier that supports probability estimates."
        )

    probabilities = model.predict_proba(processed)[0]
    classes = list(getattr(model, "classes_", []))

    if 1 not in classes:
        raise RuntimeError(
            "The model's classes do not include churn class 1. "
            "Check target encoding in train.py and retrain the model."
        )

    probability = float(probabilities[classes.index(1)])
    return max(0.0, min(1.0, probability))


def _retention_suggestions(data):
    """Generate transparent, rule-based review ideas from customer attributes.

    These are operational prompts, not causal explanations from the ML model.
    """
    suggestions = []

    contract = str(data.get("Contract", "")).strip().lower()
    tenure = data.get("tenure")
    internet_service = str(data.get("InternetService", "")).strip().lower()
    online_security = str(data.get("OnlineSecurity", "")).strip().lower()
    tech_support = str(data.get("TechSupport", "")).strip().lower()
    payment_method = str(data.get("PaymentMethod", "")).strip().lower()
    monthly_charges = data.get("MonthlyCharges")

    if contract == "month-to-month":
        suggestions.append(
            "Review whether a longer-term plan offers suitable value for this customer; "
            "do not assume they will accept a contract change."
        )

    try:
        if tenure is not None and float(tenure) < 12:
            suggestions.append(
                "Consider an early-tenure check-in to understand onboarding issues "
                "and confirm the service is meeting expectations."
            )
    except (TypeError, ValueError):
        pass

    if internet_service not in {"", "no", "none"}:
        if online_security == "no":
            suggestions.append(
                "Explain available online-security options and pricing, if relevant "
                "to the customer's needs."
            )
        if tech_support == "no":
            suggestions.append(
                "Make support options easy to find and ask whether the customer has "
                "experienced unresolved technical issues."
            )

    if "electronic check" in payment_method:
        suggestions.append(
            "Check whether the current payment method is convenient and reliable; "
            "offer alternatives only if the customer prefers them."
        )

    try:
        if monthly_charges is not None and float(monthly_charges) >= 90:
            suggestions.append(
                "Review the bill with the customer and explain the services and charges "
                "so they can assess the value they receive."
            )
    except (TypeError, ValueError):
        pass

    if not suggestions:
        suggestions.append(
            "Ask about satisfaction and unresolved issues, and continue monitoring "
            "the customer using approved service and engagement signals."
        )

    return suggestions


def predict_with_probability(data, threshold=0.50):
    """Return (label, suggestions, estimated churn probability).

    A churn label is returned when P(class=1) >= threshold.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    processed = preprocess_input(data)
    churn_probability = _get_churn_probability(processed)
    result = "YES" if churn_probability >= float(threshold) else "NO"
    suggestions = _retention_suggestions(data)
    return result, suggestions, churn_probability


def predict(data):
    """Backward-compatible interface returning (label, suggestions)."""
    result, suggestions, _ = predict_with_probability(data, threshold=0.50)
    return result, suggestions
