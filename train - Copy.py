import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

print("\n--- Dataset Shape ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- Column Names ---")
print(df.columns)

# Detect target column automatically
target_column = None
for col in df.columns:
    if "churn" in col.lower() or "exit" in col.lower() or "target" in col.lower():
        target_column = col
        break

if target_column is None:
    raise ValueError("No target column detected!")

print("\nDetected Target Column:", target_column)

# Detect column types
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# Remove ID-like columns
for col in df.columns:
    if "id" in col.lower():
        df = df.drop(col, axis=1)

print("\nRemoved ID columns")

# Convert TotalCharges if exists
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df = df.dropna()

print("\nAfter removing missing values:")
print(df.shape)

# Convert target column to numeric
df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})

# Convert categorical columns to numeric
df = pd.get_dummies(df)

print("\nFinal processed data:")
print(df.head())

print("\nTotal columns after processing:", len(df.columns))

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Feature importance
importances = model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# Sample prediction
sample = X_test.iloc[0:1]   # FIXED (2D input)
prediction = model.predict(sample)

print("\nPrediction for sample customer:")
print("Churn:", "YES" if prediction[0] == 1 else "NO")

# Suggestions
def suggest(sample_row):
    suggestions = []

    row = sample_row.iloc[0]

    if 'MonthlyCharges' in row and row['MonthlyCharges'] > 70:
        suggestions.append("Offer discount")

    if 'tenure' in row and row['tenure'] < 6:
        suggestions.append("Provide onboarding support")

    if not suggestions:
        suggestions.append("Maintain good service")

    return suggestions

suggestions = suggest(sample)

print("\nSuggestions:")
for s in suggestions:
    print("-", s)

# Visualization
import matplotlib.pyplot as plt

# Churn distribution
plt.figure()
df[target_column].value_counts().plot(kind='bar')
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Top features
top_features = feature_importance_df.head(10)

plt.figure()
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# Histogram
if 'MonthlyCharges' in df.columns:
    plt.figure()
    df['MonthlyCharges'].hist()
    plt.title("Monthly Charges Distribution")
    plt.xlabel("Charges")
    plt.ylabel("Frequency")
    plt.show()

# Save model
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("\nModel saved as model.pkl")
print("Columns saved as columns.pkl")