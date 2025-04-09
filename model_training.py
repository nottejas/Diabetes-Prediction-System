import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV
import joblib
import os
import xgboost as xgb
from data_preprocessing import preprocess_data


def train_and_evaluate_models(data_splits):
    """
    Train and evaluate multiple machine learning models for diabetes prediction

    Parameters:
    data_splits (dict): Dictionary containing split datasets and feature names

    Returns:
    dict: Dictionary containing model results
    """
    # Create directories for output
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Define the models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    # Parameters for grid search
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }

    # Train and evaluate each model
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Create and train the grid search
        grid_search = GridSearchCV(
            model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(data_splits['X_train'], data_splits['y_train'])

        # Get the best model
        best_model = grid_search.best_estimator_

        # Save the model
        joblib.dump(best_model, f'models/{name.replace(" ", "_").lower()}.pkl')

        # Make predictions on validation set
        y_val_pred = best_model.predict(data_splits['X_val'])
        y_val_prob = best_model.predict_proba(data_splits['X_val'])[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(data_splits['y_val'], y_val_pred)
        precision = precision_score(data_splits['y_val'], y_val_pred)
        recall = recall_score(data_splits['y_val'], y_val_pred)
        f1 = f1_score(data_splits['y_val'], y_val_pred)
        roc_auc = roc_auc_score(data_splits['y_val'], y_val_prob)

        # Store results
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(data_splits['y_val'], y_val_pred),
            'classification_report': classification_report(data_splits['y_val'], y_val_pred)
        }

        # Print results
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation ROC AUC: {roc_auc:.4f}")
        print(f"\nConfusion Matrix:\n{results[name]['confusion_matrix']}")
        print(f"\nClassification Report:\n{results[name]['classification_report']}")

    return results


def select_best_model(results, data_splits):
    """
    Select the best model based on ROC AUC score and evaluate it on the test set

    Parameters:
    results (dict): Dictionary containing model results
    data_splits (dict): Dictionary containing split datasets

    Returns:
    tuple: Best model and its name
    """
    # Find the model with the highest ROC AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']

    print(f"\nBest Model: {best_model_name}")
    print(f"Validation ROC AUC: {results[best_model_name]['roc_auc']:.4f}")

    # Evaluate on the test set
    y_test_pred = best_model.predict(data_splits['X_test'])
    y_test_prob = best_model.predict_proba(data_splits['X_test'])[:, 1]

    # Calculate metrics
    test_accuracy = accuracy_score(data_splits['y_test'], y_test_pred)
    test_precision = precision_score(data_splits['y_test'], y_test_pred)
    test_recall = recall_score(data_splits['y_test'], y_test_pred)
    test_f1 = f1_score(data_splits['y_test'], y_test_pred)
    test_roc_auc = roc_auc_score(data_splits['y_test'], y_test_prob)

    print(f"\nTest Metrics for {best_model_name}:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")

    print(f"\nConfusion Matrix:\n{confusion_matrix(data_splits['y_test'], y_test_pred)}")
    print(f"\nClassification Report:\n{classification_report(data_splits['y_test'], y_test_pred)}")

    # Save the best model as the default model
    joblib.dump(best_model, 'models/best_model.pkl')

    # Plot ROC curves
    plt.figure(figsize=(10, 8))

    # Compute ROC curve and ROC area for each model
    for name, result in results.items():
        model = result['model']
        y_test_prob = model.predict_proba(data_splits['X_test'])[:, 1]
        fpr, tpr, _ = roc_curve(data_splits['y_test'], y_test_prob)
        auc = roc_auc_score(data_splits['y_test'], y_test_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('visualizations/roc_curves.png')
    plt.close()

    return best_model, best_model_name


def main():
    # Preprocess data
    data_splits = preprocess_data('data/diabetes.csv')

    # Train and evaluate models
    results = train_and_evaluate_models(data_splits)

    # Select the best model
    best_model, best_model_name = select_best_model(results, data_splits)

    print(f"\nModel training completed. Best model: {best_model_name}")
    print("All models are saved in the 'models' directory.")
    print("Visualizations are saved in the 'visualizations' directory.")


if __name__ == "__main__":
    main()