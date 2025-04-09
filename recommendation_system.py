def categorize_risk(probability):
    """
    Categorize diabetes risk based on prediction probability

    Parameters:
    probability (float): Predicted probability of diabetes

    Returns:
    str: Risk category (Low, Medium, or High)
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"


def get_treatment_recommendation(risk_category, patient_data):
    """
    Generate personalized treatment recommendations based on risk category and patient data

    Parameters:
    risk_category (str): Risk category (Low, Medium, or High)
    patient_data (dict or pandas.Series): Patient's health data

    Returns:
    dict: Dictionary of recommendations by category
    """
    # Convert patient data to a dictionary if it's not already
    if hasattr(patient_data, 'to_dict'):
        patient_dict = patient_data.to_dict()
    else:
        patient_dict = patient_data

    # Base recommendations by risk category
    if risk_category == "Low Risk":
        recommendations = {
            "Lifestyle": [
                "Maintain a balanced diet with moderate carbohydrate intake",
                "Engage in regular physical activity (150 minutes per week)",
                "Maintain a healthy weight"
            ],
            "Monitoring": [
                "Annual blood glucose screening",
                "Annual check-up with healthcare provider"
            ],
            "Education": [
                "Learn about diabetes risk factors and prevention strategies"
            ]
        }

    elif risk_category == "Medium Risk":
        recommendations = {
            "Lifestyle": [
                "Follow a structured diet plan with reduced carbohydrate intake",
                "Increase physical activity to 150-300 minutes per week",
                "Aim for 5-7% weight loss if BMI > 25"
            ],
            "Monitoring": [
                "Blood glucose screening every 6 months",
                "Regular blood pressure monitoring",
                "Consider HbA1c testing"
            ],
            "Education": [
                "Diabetes prevention program enrollment",
                "Nutritional counseling"
            ]
        }

    else:  # High Risk
        recommendations = {
            "Lifestyle": [
                "Structured diet plan with careful carbohydrate counting",
                "Supervised exercise program (minimum 300 minutes per week)",
                "Target 7-10% weight loss if BMI > 25"
            ],
            "Monitoring": [
                "Regular blood glucose monitoring (consider home testing)",
                "HbA1c testing every 3 months",
                "Comprehensive metabolic panel every 6 months"
            ],
            "Medical Intervention": [
                "Consider preventive medication (e.g., Metformin)",
                "Regular consultation with endocrinologist"
            ],
            "Education": [
                "Intensive diabetes prevention program",
                "Education on recognizing symptoms of diabetes"
            ]
        }

    # Personalize recommendations based on patient data
    if 'BMI' in patient_dict and patient_dict['BMI'] >= 30:
        if risk_category == "Low Risk":
            recommendations["Lifestyle"].append("Consider a structured weight management program")
        else:
            recommendations["Lifestyle"].append("Enroll in a medically supervised weight management program")
            recommendations["Monitoring"].append("Monitor for signs of metabolic syndrome")

    if 'Age' in patient_dict and patient_dict['Age'] >= 60:
        recommendations["Monitoring"].append("Include cardiovascular risk assessment")

    if 'Glucose' in patient_dict and patient_dict['Glucose'] >= 140:
        if risk_category != "Low Risk":
            if "Medical Intervention" not in recommendations:
                recommendations["Medical Intervention"] = []
            recommendations["Medical Intervention"].append(
                "Consult healthcare provider about impaired glucose tolerance")

    if 'BloodPressure' in patient_dict and patient_dict['BloodPressure'] >= 140:
        if "Medical Intervention" not in recommendations:
            recommendations["Medical Intervention"] = []
        recommendations["Medical Intervention"].append("Blood pressure management plan")

    return recommendations


# For testing the module directly
if __name__ == "__main__":
    # Example patient data
    test_patient = {
        'Pregnancies': 2,
        'Glucose': 130,
        'BloodPressure': 80,
        'SkinThickness': 31,
        'Insulin': 95,
        'BMI': 32.4,
        'DiabetesPedigreeFunction': 0.3,
        'Age': 45
    }

    # Test with different risk categories
    for risk in ["Low Risk", "Medium Risk", "High Risk"]:
        print(f"\n{risk} Recommendations:")
        recommendations = get_treatment_recommendation(risk, test_patient)
        for category, items in recommendations.items():
            print(f"\n{category}:")
            for item in items:
                print(f"- {item}")