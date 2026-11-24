import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco_customer_churn.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove CustomerID column
df.drop('CustomerID', axis=1, inplace=True)

# Separate target column
labels = df['Churn Label']

# Remove target column from dataset
df.drop('Churn Label', axis=1, inplace=True)

# Display cleaned dataset
print("Cleaned Dataset:")
print(df.head())

# Display labels
print("\nChurn Labels:")
print(labels.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Labels shape
print("\nLabels Shape:")
print(labels.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=labels)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# 2. Numerical data distribution
df.select_dtypes(include='number').hist(
    figsize=(12, 10),
    bins=20
)

plt.suptitle("Numerical Data Distribution")
plt.tight_layout()
plt.show()

# 3. Missing values visualization
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()

print("EDA Visualization Completed Successfully!")