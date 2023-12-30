# app.py

import streamlit as st
from model_deploy import load_model, predict

def get_mental_health_condition(prediction):
    # Mapping of mental health conditions
    mental_health_mapping = {
        0: 'ADHD',
        1: 'ASD',
        2: 'Loneliness',
        3: 'MDD',
        4: 'OCD',
        5: 'PDD',
        6: 'PTSD',
        7: 'anxiety',
        8: 'bipolar',
        9: 'eating disorder',
        10: 'psychotic depression',
        11: 'sleeping disorder'
    }

    # Get the corresponding mental health condition for the predicted key
    return mental_health_mapping.get(prediction, 'Unknown')

def main():
    st.title("Mental Health Prediction")

    # Load the model
    model = load_model()

    # Create input form for each feature
    features = [
        'feeling.nervous', 'panic', 'breathing.rapidly', 'sweating',
        'trouble.in.concentration', 'having.trouble.in.sleeping',
        'having.trouble.with.work', 'hopelessness', 'anger', 'over.react',
        'change.in.eating', 'suicidal.thought', 'feeling.tired', 'close.friend',
        'social.media.addiction', 'weight.gain', 'introvert',
        'popping.up.stressful.memory', 'having.nightmares',
        'avoids.people.or.activities', 'feeling.negative',
        'trouble.concentrating', 'blaming.yourself', 'hallucinations',
        'repetitive.behavior', 'increased.energy', 'age'
    ]

    user_inputs = {}
    for feature in features:
        if feature == 'age':
            user_inputs[feature] = st.number_input(f"Age: Enter your age", min_value=0, max_value=150, step=1, value=30)
        else:
            user_inputs[feature] = st.radio(f"{feature.capitalize()}: Select 'Yes' or 'No'", ["Yes", "No"])

    if st.button("Predict"):
        # Convert user input to binary (1 for 'Yes', 0 for 'No')
        input_data = [1 if user_inputs[feature] == "Yes" else 0 if user_inputs[feature] == "No" else user_inputs[feature] for feature in features]

        # Perform prediction using the loaded model
        prediction = predict(model, [input_data])[0]

        # Get the corresponding mental health condition for the predicted key
        predicted_condition = get_mental_health_condition(prediction)

        st.write("Predicted Mental Health Condition:", predicted_condition)

if __name__ == "__main__":
    main()
