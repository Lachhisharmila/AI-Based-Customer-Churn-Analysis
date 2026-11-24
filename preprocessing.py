import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/Telco_customer_churn.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Remove unnecessary columns
df.drop(
    ['CustomerID', 'Churn Reason', 'Churn Score', 'CLTV'],
    axis=1,
    inplace=True
)

# Separate target column
labels = df['Churn Label']
df.drop('Churn Label', axis=1, inplace=True)

# Convert text columns into numbers
encoder = LabelEncoder()

for column in df.select_dtypes(include='object').columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

# Convert target labels into numbers
labels = encoder.fit_transform(labels.astype(str))

# Display results
print("Preprocessing Completed!")
print("\nProcessed Dataset:")
print(df.head())

print("\nFeatures Shape:", df.shape)
print("Labels Shape:", labels.shape)# Save features and labels together
df['Churn'] = labels

df.to_csv("data/processed_data.csv", index=False)

print("Processed data saved successfully!")


