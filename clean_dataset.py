# Comprehensive Diabetes Dataset Cleaning Pipeline
# ===============================================
# This script performs a full cleaning procedure for the diabetes dataset:
# 1. Checks for missing values and zeros in all columns
# 2. Replaces zeros with NaN where zeros are not physiologically valid
# 3. Imputes missing values using group-wise imputation
# 4. Validates the dataset quality
# 5. Saves cleaned and imputed dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os

# Set the style for visualizations
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('default')

# Set a colorblind-friendly palette
sns.set_palette("colorblind")

print("Starting diabetes dataset cleaning pipeline...")

# 1. LOAD THE DATASET
# ===================
try:
    df = pd.read_csv('F:\DiabetesPrediction\data\diabetes.csv')
    print("Loaded dataset 'diabetes.csv'")
    print(f"Dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Display basic information about the dataset
print("\nBasic dataset information:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for non-numeric columns that would cause issues with imputation
print("\nDataframe column types:")
print(df.dtypes)

# Filter out any categorical columns to avoid issues
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

print(f"\nNumeric columns: {numeric_columns}")
print(f"Categorical columns: {categorical_columns}")

# 2. DATA QUALITY ASSESSMENT
# =========================
# Check for explicit missing values (NaN)
explicit_missing = df.isnull().sum()

# Columns where zero is not a valid physiological value
zero_invalid_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Check for zero values in those columns
zero_counts = {}
zero_percentage = {}
for col in zero_invalid_columns:
    zero_counts[col] = (df[col] == 0).sum()
    zero_percentage[col] = zero_counts[col] / len(df) * 100

print("\nExplicit missing values (NaN):")
print(explicit_missing)

print("\nImplicit missing values (zeros where not physiologically valid):")
for col in zero_invalid_columns:
    print(f"{col}: {zero_counts[col]} zeros ({zero_percentage[col]:.2f}%)")

# 3. DATA CLEANING - REPLACE ZEROS WITH NaN
# =======================================
print("\nReplacing zeros with NaN where physiologically invalid...")
df_clean = df.copy()

# Replace zeros with NaN in columns where zero is not physiologically valid
for col in zero_invalid_columns:
    df_clean.loc[df_clean[col] == 0, col] = np.nan

# Check missing values after replacement
missing_after_replacement = df_clean.isnull().sum()
print("\nMissing values after replacing zeros with NaN:")
print(missing_after_replacement)

# Calculate missing percentage
missing_percentage = missing_after_replacement / len(df_clean) * 100
print("\nMissing percentage by column:")
print(missing_percentage)

# 4. OUTLIER IDENTIFICATION
# =======================
print("\nIdentifying outliers...")
# Define normal ranges based on medical knowledge
ranges = {
    'Glucose': [70, 200],  # mg/dL
    'BloodPressure': [60, 140],  # mm Hg
    'SkinThickness': [5, 50],  # mm
    'Insulin': [2, 600],  # μU/mL
    'BMI': [15, 60]  # kg/m²
}

# Identify outliers
outliers = {}
for col, (lower, upper) in ranges.items():
    outliers[col] = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].dropna()[col]
    print(f"{col} outliers (outside {lower}-{upper}): {len(outliers[col])} values")
    if len(outliers[col]) > 0:
        print(f"  Min: {outliers[col].min()}, Max: {outliers[col].max()}")

# 5. VISUALIZE DISTRIBUTIONS BEFORE IMPUTATION
# =========================================
print("\nCreating pre-imputation visualizations...")


# Function to create and save distribution plots
def plot_distributions(data, cols, filename_prefix="pre_imputation"):
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(cols):
        plt.subplot(3, 2, i + 1)
        sns.histplot(data[data[col] > 0][col], kde=True)
        plt.title(f'Distribution of {col} (Excluding Zeros)')
        plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_distributions.png')
    plt.close()


# Create distribution plots before imputation
try:
    plot_distributions(df_clean, zero_invalid_columns)
    print("Saved pre-imputation distribution plots.")
except Exception as e:
    print(f"Warning: Could not create distribution plots: {e}")

# 6. EXAMINE RELATIONSHIPS BY OUTCOME FOR IMPUTATION STRATEGY
# ========================================================
print("\nExamining relationships by diabetes outcome...")

# Create boxplots to compare values by outcome
try:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(zero_invalid_columns):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(data=df_clean, x='Outcome', y=col)
        plt.title(f'{col} by Diabetes Outcome')
        plt.xlabel('Outcome (0=Non-Diabetic, 1=Diabetic)')
    plt.tight_layout()
    plt.savefig('variables_by_outcome.png')
    plt.close()
    print("Saved boxplots of variables by outcome.")
except Exception as e:
    print(f"Warning: Could not create boxplots: {e}")

# 7. IMPUTE MISSING VALUES
# =======================
print("\nImputing missing values...")

# Group-wise imputation (by Outcome)
df_imputed = df_clean.copy()

# Create a copy for comparison later
df_before_imputation = df_clean.copy()

# We'll only impute numeric columns
cols_with_missing = [col for col in numeric_columns
                     if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0]
print(f"Columns with missing values to impute: {cols_with_missing}")

# Track imputation statistics for reporting
imputation_stats = {}

# Impute by diabetes outcome group
for outcome in [0, 1]:
    group_name = "Diabetic" if outcome == 1 else "Non-Diabetic"
    print(f"\nImputing values for {group_name} group...")

    for col in cols_with_missing:
        # Calculate the median for this outcome group
        group_subset = df_imputed[df_imputed['Outcome'] == outcome][col]
        if group_subset.notnull().sum() > 0:
            group_median = group_subset.median()

            # Count how many values will be imputed
            to_impute_count = df_imputed[(df_imputed['Outcome'] == outcome) &
                                         (df_imputed[col].isnull())].shape[0]

            # Apply the group median to missing values in this group
            df_imputed.loc[(df_imputed['Outcome'] == outcome) &
                           (df_imputed[col].isnull()), col] = group_median

            # Store imputation statistics
            if col not in imputation_stats:
                imputation_stats[col] = {0: {"count": 0, "value": None},
                                         1: {"count": 0, "value": None}}

            imputation_stats[col][outcome]["count"] = to_impute_count
            imputation_stats[col][outcome]["value"] = group_median

            print(f"  {col}: Imputed {to_impute_count} values with median {group_median:.2f}")
        else:
            print(f"  {col}: No non-null values in {group_name} group for imputation")

# 8. VERIFY NO MISSING VALUES REMAIN
# =================================
missing_after_imputation = df_imputed.isnull().sum()
print("\nMissing values after imputation:")
print(missing_after_imputation)

if missing_after_imputation.sum() > 0:
    print("\nWARNING: Some missing values could not be imputed by group-wise approach.")
    print("Applying standard median imputation for any remaining missing values...")

    # Get columns with remaining missing values
    remaining_cols_with_missing = [col for col in cols_with_missing
                                   if df_imputed[col].isnull().sum() > 0]

    if remaining_cols_with_missing:
        # Apply standard median imputation for any remaining missing values
        median_imputer = SimpleImputer(strategy='median')
        df_imputed[remaining_cols_with_missing] = median_imputer.fit_transform(
            df_imputed[remaining_cols_with_missing])

        # Verify again
        missing_after_standard = df_imputed[remaining_cols_with_missing].isnull().sum()
        print("Missing values after standard imputation:")
        print(missing_after_standard)

# 9. VISUALIZE DISTRIBUTIONS AFTER IMPUTATION
# ========================================
print("\nCreating post-imputation visualizations...")


# Function to compare distributions before and after imputation
def compare_distributions(before_df, after_df, cols, filename="imputation_comparison"):
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(cols):
        plt.subplot(3, 2, i + 1)
        # Original non-zero values
        sns.kdeplot(before_df[before_df[col] > 0][col], label='Original (non-zero)', linewidth=2)
        # Imputed values
        sns.kdeplot(after_df[col], label='After Imputation', linewidth=2, alpha=0.7)
        plt.title(f'Distribution of {col} Before and After Imputation')
        plt.xlabel(col)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


try:
    compare_distributions(df_before_imputation, df_imputed, zero_invalid_columns)
    print("Saved distribution comparison plots.")
except Exception as e:
    print(f"Warning: Could not create comparison plots: {e}")

# 10. SUMMARIZE CHANGES
# ===================
print("\nCreating summary of changes...")

# Create a DataFrame to display the summary
summary = pd.DataFrame(columns=['Column', 'Missing Before', 'Missing After',
                                'Original Mean', 'Imputed Mean',
                                'Original Median', 'Imputed Median'])

row_counter = 0
for col in cols_with_missing:
    try:
        # Calculate statistics for non-zero values in original data
        original_stats = df[df[col] > 0][col].describe()
        imputed_stats = df_imputed[col].describe()

        # Get diabetic/non-diabetic imputation values
        diabetic_value = imputation_stats.get(col, {}).get(1, {}).get("value", "N/A")
        nondiabetic_value = imputation_stats.get(col, {}).get(0, {}).get("value", "N/A")
        diabetic_value = f"{diabetic_value:.2f}" if isinstance(diabetic_value, (int, float)) else diabetic_value
        nondiabetic_value = f"{nondiabetic_value:.2f}" if isinstance(nondiabetic_value,
                                                                     (int, float)) else nondiabetic_value

        summary.loc[row_counter] = [
            col,
            f"{df_clean[col].isnull().sum()} ({df_clean[col].isnull().sum() / len(df_clean) * 100:.2f}%)",
            f"{df_imputed[col].isnull().sum()} ({df_imputed[col].isnull().sum() / len(df_imputed) * 100:.2f}%)",
            f"{original_stats['mean']:.2f}",
            f"{imputed_stats['mean']:.2f}",
            f"{original_stats['50%']:.2f}",
            f"{imputed_stats['50%']:.2f}"
        ]
        row_counter += 1
    except Exception as e:
        print(f"Error summarizing column {col}: {e}")

print("\nSummary of changes after imputation:")
print(summary)

# 11. SAVE THE FULLY IMPUTED DATASET
# ================================
print("\nSaving the fully imputed dataset...")
output_filename = 'diabetes_complete.csv'
df_imputed.to_csv(output_filename, index=False)
print(f"Fully imputed dataset saved to '{output_filename}'")
print(f"File saved at: {os.path.abspath(output_filename)}")

# 12. FINAL DATA QUALITY CHECK
# ===========================
print("\nPerforming final data quality check...")

print("Final statistics for key columns:")
for col in zero_invalid_columns:
    min_val = df_imputed[col].min()
    max_val = df_imputed[col].max()
    mean_val = df_imputed[col].mean()
    std_val = df_imputed[col].std()
    print(f"{col}: Min={min_val:.1f}, Max={max_val:.1f}, Mean={mean_val:.2f}, Std={std_val:.2f}")

# Create a final correlation heatmap
try:
    # Calculate correlations
    correlation = df_imputed[numeric_columns].corr()

    # Create and save heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Variables (After Imputation)')
    plt.tight_layout()
    plt.savefig('final_correlation_heatmap.png')
    plt.close()
    print("Saved final correlation heatmap.")
except Exception as e:
    print(f"Warning: Could not create correlation heatmap: {e}")

# Create a pairplot for key variables
try:
    # Select key variables for the pairplot
    plot_vars = ['Glucose', 'BMI', 'Insulin', 'Age', 'Outcome']
    plot_vars = [var for var in plot_vars if var in df_imputed.columns]

    # Create and save pairplot
    sns.pairplot(df_imputed, vars=plot_vars[:-1], hue='Outcome')
    plt.suptitle('Relationships Between Key Variables After Imputation', y=1.02)
    plt.savefig('final_pairplot.png')
    plt.close()
    print("Saved final pairplot.")
except Exception as e:
    print(f"Warning: Could not create pairplot: {e}")

print("\nData cleaning and imputation pipeline completed successfully!")
print("=============================================================")
print("\nSummary of processed dataset:")
print(f"Total records: {len(df_imputed)}")
print(f"Columns: {', '.join(df_imputed.columns)}")
print(f"Missing values: {df_imputed.isnull().sum().sum()}")
print("\nYou can now use the cleaned dataset 'diabetes_complete.csv' for analysis and modeling.")