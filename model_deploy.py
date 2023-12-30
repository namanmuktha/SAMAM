# model_deploy.py

import joblib

def load_model():
    # Load your trained model
    model = joblib.load("model.pkl")
    return model

def predict(model, input_data):
    # Perform predictions using the loaded model
    predictions = model.predict(input_data)
    return predictions
