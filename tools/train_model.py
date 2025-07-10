import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save():
    file_path = "data/pima_diabetes.csv"  # local file with header row
    df = pd.read_csv(file_path)  # remove header=None

    # Select columns
    X = df[['Age', 'BMI', 'DiabetesPedigreeFunction']]
    y = df['Outcome']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_and_save()
