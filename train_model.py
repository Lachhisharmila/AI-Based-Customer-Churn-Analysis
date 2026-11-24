import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load original dataset
df = pd.read_csv("data/Telco_customer_churn.csv")

df.columns = df.columns.str.strip()

# Select useful customer features
features = [
    'Gender',
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Tenure Months',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method',
    'Monthly Charges',
    'Total Charges'
]

# Prepare features and target
X = df[features].copy()
y = df['Churn Value']

# Clean numeric columns
numeric_features = [
    'Tenure Months',
    'Monthly Charges',
    'Total Charges'
]

categorical_features = [
    col for col in features if col not in numeric_features
]

X['Total Charges'] = pd.to_numeric(
    X['Total Charges'], errors='coerce'
)

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create complete model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Model Training Completed!")
print("Accuracy:", accuracy_score(y_test, predictions) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "data/churn_model.pkl")

print("\nNew model saved successfully!")
