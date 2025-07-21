import joblib
import numpy as np

# Load the trained model from disk once at import
model = joblib.load("models/model.pkl")


def predict_diabetes_risk(age: float, bmi: float, diabetes_pedigree_function: float) -> dict:
    
    X = np.array([[age, bmi, diabetes_pedigree_function]])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # Probability of class 1 (diabetes)
    return {
        "prediction": int(prediction),
        "probability": round(float(proba), 4)
    }