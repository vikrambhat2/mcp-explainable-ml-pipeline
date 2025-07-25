import joblib
import shap
import pandas as pd

# Load the trained model (only once)
model = joblib.load("models/model.pkl")

# Create SHAP explainer globally
explainer = shap.TreeExplainer(model)

def explain_diabetes_risk(age: float, bmi: float, diabetes_pedigree_function: float) -> dict:
    """
    Explains the prediction of diabetes risk using SHAP values.

    This function computes SHAP (SHapley Additive exPlanations) values to show how each input feature 
    contributes to the model's prediction for diabetes risk.

    Args:
        age (float): Age of the individual.
        bmi (float): Body Mass Index.
        diabetes_pedigree_function (float): A measure of hereditary diabetes risk.

    Returns:
        dict: A dictionary mapping each feature to a pair of SHAP values:
              [contribution to class 0 (no diabetes), contribution to class 1 (diabetes)].
              Positive SHAP values for class 1 indicate that the feature increases predicted diabetes risk.
    """
    # Wrap input into DataFrame for SHAP
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "diabetes_pedigree_function": diabetes_pedigree_function
    }])

    # Compute SHAP values (returns list of arrays for classification)
    shap_values = explainer.shap_values(input_df)

    # Use SHAP values for class 1 (positive class: diabetic)
    explanation = {k: shap_values[0].tolist()[i][1] for i, k in enumerate(input_df.columns)}  # ← safe serialization


    return {
        "explanation": explanation
    }