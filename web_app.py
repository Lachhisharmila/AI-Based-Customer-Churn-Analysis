import os
import pandas as pd
import joblib

from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load saved model
folder = os.path.dirname(os.path.abspath(__file__))

model_data = joblib.load(
    os.path.join(folder, "churn_model.pkl")
)

model = model_data["model"]
scaler = model_data["scaler"]
feature_columns = model_data["feature_columns"]
uses_scaling = model_data["uses_scaling"]

# Load dataset for customer input preparation
file_path = os.path.join(
    folder, "data", "Telco_customer_churn.csv"
)

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Prepare target
target_col = "Churn Label"

y = (
    df[target_col]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

df = df.loc[y.notna()].copy()

# Prepare input columns
X_raw = df.drop(columns=[target_col])

remove_cols = [
    col for col in X_raw.columns
    if col.lower().replace(" ", "").replace("_", "")
    in [
        "customerid", "churnvalue", "churnscore",
        "churnreason", "churncategory", "customerstatus"
    ]
    or "churn" in col.lower()
]

X_raw = X_raw.drop(columns=remove_cols, errors="ignore")

# Handle missing values
for col in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[col]):
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    else:
        mode = X_raw[col].mode()
        X_raw[col] = X_raw[col].fillna(
            mode.iloc[0] if not mode.empty else "Unknown"
        )

# Find dataset columns
def find_column(names):
    for col in X_raw.columns:
        normalized = col.lower().replace(" ", "").replace("_", "")
        if normalized in names:
            return col
    return None

tenure_col = find_column(["tenure", "tenuremonths"])
monthly_col = find_column(["monthlycharges"])
contract_col = find_column(["contract"])

# Default customer details
default_row = {}

for col in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[col]):
        default_row[col] = X_raw[col].median()
    else:
        default_row[col] = X_raw[col].mode().iloc[0]

contract_options = []

if contract_col:
    contract_options = sorted(
        X_raw[contract_col].dropna().astype(str).unique().tolist()
    )

# Website page
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Customer Churn Prediction</title>
    <style>
        body {
            font-family: Arial;
            background: #edf2f7;
            text-align: center;
            padding: 30px;
        }
        .container {
            background: white;
            max-width: 480px;
            margin: auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px #cccccc;
        }
        h1 { color: #173b65; }
        input, select {
            width: 90%;
            padding: 12px;
            margin: 10px;
            border: 1px solid #bbb;
            border-radius: 6px;
        }
        button {
            background: #1769aa;
            color: white;
            padding: 13px 30px;
            border: none;
            border-radius: 7px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #e8f4ff;
            border-radius: 8px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Customer Churn Prediction</h1>
    <p>Enter customer details below</p>

    <form method="POST">
        {% if tenure_col %}
        <label>Tenure (Months)</label><br>
        <input type="number" name="tenure"
               min="0" max="120" value="12" required>
        {% endif %}

        {% if monthly_col %}
        <label>Monthly Charges</label><br>
        <input type="number" name="monthly"
               min="0" step="0.01" value="70" required>
        {% endif %}

        {% if contract_col %}
        <label>Contract Type</label><br>
        <select name="contract">
            {% for option in contract_options %}
            <option value="{{ option }}">{{ option }}</option>
            {% endfor %}
        </select>
        {% endif %}

        <br><br>
        <button type="submit">Predict Churn</button>
    </form>

    {% if result %}
    <div class="result">
        <h2>{{ result }}</h2>
        <h3>Churn Probability: {{ probability }}%</h3>
        <p>Model: {{ model_name }}</p>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None

    if request.method == "POST":

        customer = default_row.copy()

        if tenure_col:
            customer[tenure_col] = float(request.form["tenure"])

        if monthly_col:
            customer[monthly_col] = float(request.form["monthly"])

        if contract_col:
            selected_contract = request.form["contract"]
            if selected_contract not in contract_options:
                return "Invalid contract type", 400
            customer[contract_col] = selected_contract

        # Encode customer details consistently
        combined = pd.concat(
            [X_raw, pd.DataFrame([customer])],
            ignore_index=True
        )

        encoded = pd.get_dummies(
            combined, drop_first=True, dtype=int
        )

        input_data = encoded.tail(1).reindex(
            columns=feature_columns,
            fill_value=0
        )

        if uses_scaling:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = round(
                model.predict_proba(input_data)[0][1] * 100, 2
            )
        else:
            probability = 0

        if prediction == 1:
            result = "Customer is likely to Churn!"
        else:
            result = "Customer is likely to Stay!"

    return render_template_string(
        HTML,
        result=result,
        probability=probability,
        model_name=model_data["model_name"],
        tenure_col=tenure_col,
        monthly_col=monthly_col,
        contract_col=contract_col,
        contract_options=contract_options
    )


if __name__ == "__main__":
    app.run(debug=True)