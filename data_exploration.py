import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Load the dataset
data_path = 'data/diabetes.csv'
data = pd.read_csv(data_path)
print(data)


# Display basic information
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nDescriptive Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check for zero values in features where zeros don't make sense
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_columns:
    print(f"Zeros in {column}: {(data[column] == 0).sum()}")

# Create visualizations

# 1. Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png')
plt.close()

# 2. Distribution of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=data)
plt.title('Distribution of Diabetes Outcome')
plt.ylabel('Count')
plt.savefig('visualizations/outcome_distribution.png')
plt.close()

# 3. Feature distributions by outcome
for feature in data.columns[:-1]:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=feature, hue='Outcome', kde=True)
    plt.title(f'Distribution of {feature} by Diabetes Outcome')
    plt.tight_layout()
    plt.savefig(f'visualizations/{feature}_distribution.png')
    plt.close()

# 4. Boxplots for each feature by outcome
for feature in data.columns[:-1]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y=feature, data=data)
    plt.title(f'Boxplot of {feature} by Diabetes Outcome')
    plt.tight_layout()
    plt.savefig(f'visualizations/{feature}_boxplot.png')
    plt.close()

print("Data exploration completed. Visualizations saved to 'visualizations' directory.")