import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("===== AI CUSTOMER CHURN PREDICTION =====")

# Step 1: Load Dataset
folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "data", "Telco_customer_churn.csv")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

print("\nDataset Loaded:", df.shape)

# Step 2: Clean Dataset
df = df.drop_duplicates()
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Step 3: Identify Target Column
target_col = "Churn Label"

if target_col not in df.columns:
    raise ValueError("Churn Label column not found!")

# Step 4: Convert Yes/No into 1/0
y = (
    df[target_col]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

valid = y.notna()
df = df.loc[valid].copy()
y = y.loc[valid].astype(int)

print("\nChurn Values Converted Successfully!")

# Step 5: Prepare Features
X = df.drop(columns=[target_col])

# Remove ID and information that reveals the target
remove_cols = [
    col for col in X.columns
    if col.lower().replace(" ", "").replace("_", "")
    in [
        "customerid", "churnvalue", "churnscore",
        "churnreason", "churncategory", "customerstatus"
    ]
    or "churn" in col.lower()
]

X = X.drop(columns=remove_cols, errors="ignore")

# Step 6: Handle Missing Values
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        mode = X[col].mode()
        X[col] = X[col].fillna(
            mode.iloc[0] if not mode.empty else "Unknown"
        )

# Step 7: Convert Categorical Values
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nFeatures Prepared:", X.shape)

# Step 8: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 9: Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Logistic Regression
print("\n===== LOGISTIC REGRESSION =====")

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Accuracy:", round(lr_accuracy * 100, 2), "%")
print(classification_report(
    y_test, lr_predictions, zero_division=0
))

# Step 11: Random Forest
print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Accuracy:", round(rf_accuracy * 100, 2), "%")
print(classification_report(
    y_test, rf_predictions, zero_division=0
))

# Step 12: Model Comparison
print("\n===== MODEL COMPARISON =====")

print("Logistic Regression:",
      round(lr_accuracy * 100, 2), "%")

print("Random Forest:",
      round(rf_accuracy * 100, 2), "%")

print("\n===== MODEL TRAINING COMPLETED! =====")
import joblib

# Step 13: Select Best Model
if rf_accuracy >= lr_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = lr_model
    best_model_name = "Logistic Regression"

# Step 14: Save Model
model_data = {
    "model": best_model,
    "model_name": best_model_name,
    "scaler": scaler,
    "feature_columns": X.columns.tolist(),
    "uses_scaling": best_model_name == "Logistic Regression"
}

model_path = os.path.join(folder, "churn_model.pkl")

joblib.dump(model_data, model_path)

print("\n===== MODEL SAVED SUCCESSFULLY =====")
print("Best Model:", best_model_name)
print("Saved Location:", model_path)