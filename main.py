import os
import sys


def check_dataset():
    """Check if the dataset is available"""
    if not os.path.exists('data/diabetes.csv'):
        print("Error: Dataset not found at 'data/diabetes.csv'")
        print("Please download the dataset from Kaggle and place it in the 'data' directory.")
        return False
    return True


def main():
    """Main function to run the entire workflow"""
    # Check if the dataset is available
    if not check_dataset():
        return

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Step 1: Data exploration
    print("\n===== Step 1: Data Exploration =====")
    import data_exploration

    # Step 2: Data preprocessing
    print("\n===== Step 2: Data Preprocessing =====")
    from data_preprocessing import preprocess_data
    data_splits = preprocess_data('data/diabetes.csv')

    # Step 3: Model training and evaluation
    print("\n===== Step 3: Model Training and Evaluation =====")
    from model_training import train_and_evaluate_models, select_best_model
    results = train_and_evaluate_models(data_splits)
    best_model, best_model_name = select_best_model(results, data_splits)

    print("\n===== All Steps Completed Successfully =====")
    print("\nYou can now run the web application using:")
    print("python app.py")


if __name__ == "__main__":
    main()