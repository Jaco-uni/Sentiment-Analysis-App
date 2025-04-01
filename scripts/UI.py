import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the src directory to sys.path
from src import config
import streamlit as st
import pickle

# Load the models and vectorizer
with open(os.path.join(config.MODELS_PATH, "random_forest.pickle"), "rb") as file:
    modelRF = pickle.load(file)

with open(f"{config.MODELS_PATH}vectorizerRF.pickle", "rb") as f:
        vectorizerRF = pickle.load(f)

with open(f"{config.MODELS_PATH}vectorizerLR.pickle", "rb") as f:
        vectorizerLR = pickle.load(f)

with open(os.path.join(config.MODELS_PATH, "logistic_regression.pickle"), "rb") as file:
    modelLR = pickle.load(file)

selectedModel = st.selectbox(
         'Quale modello vorresti utilizzare per la sentiment analysis?',
         ('Random Forest', 'Logistic Regression'))

st.title("Text Classification")

# Text input
user_input = st.text_area("enter text to classify", "")

# Predict when button is clicked
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input and predict
        if selectedModel == "Random Forest":
            st.write("Hai selezionato Random Forest")
            x = vectorizerRF.transform([user_input]) 
            prediction = modelRF.predict(x)[0]
        
        elif selectedModel == "Logistic Regression":
            x = vectorizerLR.transform([user_input])
            prediction = modelLR.predict(x)[0]             
        if prediction == "positive":
            st.success(f"Predicted class: {prediction}")
            color="green"
        elif prediction == "neutral":
            st.success(f"Predicted class: {prediction}") #aggiungo neutral in quanto da problemi a livello di visualizzazione per il random forest
            color="blue"
        elif prediction == 'negative':
            st.warning(f"Predicted class: {prediction}")
            color="red"

import streamlit as st
import random
import time

# Funzione per generare la pioggia di emoji
def rain_emojis():
    # Impostiamo il numero di emoji e la velocità
    num_emojis = 50
    max_height = 400  # Altezza massima delle emoji
    interval = 0.05  # Intervallo tra le emoji
    speed = 10  # Velocità della pioggia (numero di emoji per secondo)
    
    for _ in range(num_emojis):
        # Posizione casuale per l'emoji
        x_pos = random.randint(0, 100)
        y_pos = random.randint(0, max_height)

        
        # Colore in base all'emoji
        if prediction == 'negative':
            emoji = "😞"
            color = "red"
        else:
            emoji= "😊"
            color = "green"
        
        # Stampa l'emoji con il suo colore e posizione
        st.markdown(
            f'<p style="position: absolute; top: {y_pos}px; left: {x_pos}%; font-size: 30px; color: {color};">{emoji}</p>',
            unsafe_allow_html=True
        )
        
        # Pausa per creare l'effetto di pioggia
        time.sleep(interval)

# Eseguiamo l'effetto pioggia
rain_emojis()
