import os
import logging
from typing import Dict, Any
from fastmcp import FastMCP
from predict import predict_diabetes_risk
import json 

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

try:
    mcp = FastMCP("Predict Diabetes Server")
except Exception as e:
    logging.error(f"Failed to initialize FastMCP: {e}")
    raise

@mcp.tool()
def diabetes_risk_predictor(age: float, bmi: float, diabetes_pedigree_function: float) -> dict:
    """Predict diabetes risk based on age, BMI, and diabetes pedigree function.
    
    Returns a dictionary with prediction (0 or 1) and probability (0-1).
    """
    return predict_diabetes_risk(age, bmi, diabetes_pedigree_function)


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
        mcp.run()
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise