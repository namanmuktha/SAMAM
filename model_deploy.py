# model_deploy.py

import pickle

def load_model():
    # Load your trained model using pickle
    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, input_data):
    # Perform predictions using the loaded model
    predictions = model.predict(input_data)
    return predictions
