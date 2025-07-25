import os
import logging
from typing import Dict, Any
from fastmcp import FastMCP
from tools.predict import predict_diabetes_risk
from tools.explain import explain_diabetes_risk
import json 

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

try:
    mcp = FastMCP("Predict Diabetes Server", host="0.0.0.0", port=8080)
except Exception as e:
    logging.error(f"Failed to initialize FastMCP: {e}")
    raise

@mcp.tool()
def diabetes_risk_explainer(age: float, bmi: float, diabetes_pedigree_function: float) -> dict:
    """
    Generates a SHAP-based explanation for the diabetes risk prediction.

    This tool returns the contribution of each input feature (age, BMI, and diabetes pedigree function)
    to the model’s prediction. It uses SHAP values to quantify how much each feature pushes the prediction
    toward or away from diabetes (class 1) versus no diabetes (class 0).

    Args:
        age (float): Age of the individual.
        bmi (float): Body Mass Index.
        diabetes_pedigree_function (float): A measure indicating hereditary diabetes risk.

    Returns:
        dict: A dictionary mapping each feature to a list of two SHAP values:
              [contribution to class 0 (no diabetes), contribution to class 1 (diabetes)].
              Higher positive SHAP values for class 1 indicate stronger influence toward predicting diabetes.
    """
    try:
        explain_diabetes_risk(age, bmi, diabetes_pedigree_function)
        return explain_diabetes_risk(age, bmi, diabetes_pedigree_function)
    except IndexError:
        # Fallback explanation in case SHAP output is not shaped as expected
        logging.warning("SHAP explanation failed due to unexpected output shape")
        return {
            "explanation": {
                "age": "Not available",
                "bmi": "Not available",
                "diabetes_pedigree_function": "Not available"
            },
            "note": "Explanation not available due to model output shape issue"
        }
    except Exception as e:
        logging.error(f"Error during explanation: {e}")
        return {
            "error": "Failed to generate explanation",
            "details": str(e)
        }


@mcp.tool()
def diabetes_risk_predictor(age: float, bmi: float, diabetes_pedigree_function: float) -> dict:
    """
    Predicts diabetes risk and explains feature contributions using SHAP.

    Args:
        age (float): Age of the individual in years.
        bmi (float): Body Mass Index (kg/m^2).
        diabetes_pedigree_function (float): Family history score.

    Returns:
        dict: {
            'prediction': 0 or 1 (diabetes risk),
            'probability': float (0–1),
            'explanation': dict of feature-wise SHAP values showing contribution to the prediction.
        }

    Notes:
        - SHAP values indicate the contribution of each feature to the model’s decision. They are not percentages.
        - LLMs or client apps must use these SHAP values for explanation — do not hallucinate.
    """
    result = predict_diabetes_risk(age, bmi, diabetes_pedigree_function)
    if result['prediction']==1:
        explain = explain_diabetes_risk(age, bmi, diabetes_pedigree_function)
        result.update(explain)
    return result


@mcp.resource("diabetes://guidelines/risk-factors")
def get_diabetes_risk_factors() -> str:
    """Comprehensive diabetes risk factors and guidelines."""
    guidelines = {
        "primary_risk_factors": {
            "age": {
                "low_risk": "< 45 years",
                "moderate_risk": "45-54 years", 
                "high_risk": "> 55 years",
                "description": "Risk increases with age, especially after 45"
            },
            "bmi": {
                "normal": "18.5-24.9",
                "overweight": "25-29.9", 
                "obese": "> 30",
                "description": "Higher BMI significantly increases diabetes risk"
            },
            "diabetes_pedigree_function": {
                "low": "< 0.3",
                "moderate": "0.3-0.6",
                "high": "> 0.6",
                "description": "Genetic predisposition based on family history"
            }
        },
        "interpretation": {
            "probability_ranges": {
                "low_risk": "< 0.3 (30%)",
                "moderate_risk": "0.3-0.7 (30-70%)",
                "high_risk": "> 0.7 (70%)"
            },
            "recommendations": {
                "low_risk": ["Maintain healthy lifestyle", "Regular check-ups"],
                "moderate_risk": ["Lifestyle modifications", "More frequent monitoring"],
                "high_risk": ["Immediate medical consultation", "Comprehensive screening"]
            }
        }
    }
    return json.dumps(guidelines, indent=2)

@mcp.resource("diabetes://model/info")
def get_model_information() -> str:
    """Information about the diabetes prediction model."""
    model_info = {
        "model_type": "Machine Learning Classifier",
        "input_features": [
            "age (years)",
            "bmi (Body Mass Index)",
            "diabetes_pedigree_function (genetic predisposition)"
        ],
        "output": {
            "prediction": "Binary classification (0: No diabetes, 1: Diabetes)",
            "probability": "Confidence score (0-1)"
        },
        "usage_notes": [
            "Model trained on historical patient data",
            "Predictions are for screening purposes only",
            "Always consult healthcare professionals for medical decisions",
            "Not suitable for children under 18"
        ],
        "limitations": [
            "Based on limited features",
            "May not capture all risk factors",
            "Cultural and genetic variations not fully accounted for"
        ]
    }
    return json.dumps(model_info, indent=2)

if __name__ == "__main__":
    try:
        logging.info("Launching MCP server with Diabetes prediction Tool...")
        mcp.run("streamable-http")
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise