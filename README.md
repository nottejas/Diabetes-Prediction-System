# Diabetes Prediction System

A comprehensive machine learning system for predicting diabetes risk based on medical indicators and providing personalized recommendations.

## Team Members
- **Tejas Padmakar (A046)**
- **Rugved Kharde (A029)**
- **Manav Mangoda (A036)**

## Features

- **Predictive Model**: XGBoost-based classification model to predict diabetes risk
- **Data Preprocessing**: Advanced cleaning pipeline with group-wise median imputation
- **Risk Assessment**: Categorization of risk levels based on prediction probabilities
- **Recommendation System**: Personalized medical advice based on patient characteristics
- **Web Interface**: User-friendly Flask application for healthcare providers
- **Data Visualization**: Comprehensive visualizations for model interpretation

## Project Structure

```
├── app.py                        # Flask web application
├── clean_dataset.py              # Comprehensive data cleaning pipeline
├── data/                         # Data directory
│   └── diabetes.csv              # Original dataset
├── data_exploration.py           # Exploratory data analysis
├── data_preprocessing.py         # Data preprocessing for model training
├── diabetes_complete.csv         # Cleaned dataset
├── DiabetesPrediction_Report.md  # Project report
├── main.py                       # Project entry point
├── model_training.py             # Model training and evaluation
├── models/                       # Saved models directory
├── models.py                     # Database models for web application
├── recommendation_system.py      # System for generating recommendations
├── requirements.txt              # Project dependencies
├── static/                       # Static assets for web application
├── templates/                    # HTML templates
└── visualizations/               # Visualization outputs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DiabetesPrediction.git
cd DiabetesPrediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation and Model Training

1. Run the data cleaning pipeline:
```bash
python clean_dataset.py
```

2. Train the prediction model:
```bash
python model_training.py
```

### Web Application

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:8080/
```

## Data Preprocessing

Our approach to preprocessing includes:

1. **Missing Value Handling**: Replacement of physiologically impossible zero values with NaN
2. **Outlier Detection**: Using IQR method to identify and cap outliers
3. **Group-wise Imputation**: Using median values based on diabetes outcome to preserve statistical differences
4. **Feature Scaling**: Standard scaling of numerical features

## Model Details

The XGBoost model is optimized using grid search with the following parameters:
- Number of estimators: 100, 200
- Learning rate: 0.01, 0.1
- Maximum tree depth: 3, 5

Key model performance metrics:
- Accuracy: 95%
- Precision: 94%
- Recall: 92%
- F1-Score: 93%
- ROC AUC: 97%

## Visualizations

The project includes various visualizations to aid in understanding:
- Distribution plots for each feature
- Boxplots showing feature differences by diabetes outcome
- Correlation matrix showing feature relationships
- ROC curve for model performance
- Confusion matrix for classification results
- Feature importance chart

## Recommendation System

The recommendation system provides guidance based on:
- Risk category (Low, Moderate, High)
- Patient-specific characteristics
- Medical best practices

## License

This project is for educational purposes only.

## Acknowledgments

- The Pima Indians Diabetes Dataset providers
- XGBoost and scikit-learn libraries
- Flask framework 