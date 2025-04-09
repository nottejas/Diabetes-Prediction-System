import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


def preprocess_data(data_path, test_size=0.2, validation_size=0.25, random_state=42):
    """
    Preprocess the diabetes dataset:
    - Handle missing values (zeros in some columns)
    - Feature scaling
    - Split data into train, validation, and test sets

    Parameters:
    data_path (str): Path to the diabetes.csv file
    test_size (float): Portion of data to use for testing
    validation_size (float): Portion of data to use for validation (from training set)
    random_state (int): Random seed for reproducibility

    Returns:
    dict: Dictionary containing split datasets and feature names
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load the dataset
    data = pd.read_csv(data_path)

    # Replace zeros with NaN in columns where zeros don't make sense
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        data[column] = data[column].replace(0, np.nan)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data.drop('Outcome', axis=1)),
        columns=data.drop('Outcome', axis=1).columns
    )

    # Save the imputer for future use
    joblib.dump(imputer, 'models/imputer.pkl')

    # Add the target variable back
    data_imputed['Outcome'] = data['Outcome'].values

    # Split features and target
    X = data_imputed.drop('Outcome', axis=1)
    y = data_imputed['Outcome']

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_val
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for future use
    joblib.dump(scaler, 'models/scaler.pkl')

    # Create a dictionary with the data splits
    data_splits = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': X.columns.tolist()
    }

    print("Data preprocessing completed.")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return data_splits


# For testing the module directly
if __name__ == "__main__":
    data_splits = preprocess_data('data/diabetes.csv')