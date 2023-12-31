import streamlit as st
import joblib

# Load the model
model = joblib.load('atm.pkl')

def main():
    model=load_model()