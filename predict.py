import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load columns automatically (no hardcoding)
columns = pickle.load(open("columns.pkl", "rb"))


def preprocess_input(data):
    df = pd.DataFrame([data])

    # Convert categorical to numeric
    df = pd.get_dummies(df)

    # Ensure all columns match training
    for col in columns:
        if col not in df:
            df[col] = 0

    # Remove extra columns (important!)
    df = df[columns]

    return df


def predict(data):
    processed = preprocess_input(data)
    prediction = model.predict(processed)

    result = "YES" if prediction[0] == 1 else "NO"

    # Generic suggestions (not telecom-specific)
    suggestions = []

    for key, value in data.items():
        if isinstance(value, (int, float)):
            if value > 70:
                suggestions.append(f"{key} is high → consider improvement")
            elif value < 5:
                suggestions.append(f"{key} is low → increase engagement")

    if not suggestions:
        suggestions.append("Maintain current performance")

    return result, suggestions


# Test
if __name__ == "__main__":
    sample = {
        "tenure": 2,
        "MonthlyCharges": 80,
        "TotalCharges": 150,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

    result, suggestions = predict(sample)

    print("Prediction:", result)
    print("Suggestions:", suggestions)