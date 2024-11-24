import streamlit as st
import pyautogui
import train
import numpy as np

st.title("Disease Diagnosis")

def refresh():
    pyautogui.hotkey("ctrl", "F5")

with st.sidebar:
    with st.expander("Identifiable Diseases"):
        st.write(
            "1.Psoriasis  2.Varicose Veins  3.Typhoid \n 4.Chicken pox 5.Impetigo 6.Dengue \n"
            "7.Fungal infection 8.Common Cold 9.Pneumonia \n 10.Dimorphic Hemorrhoids 11.Arthritis 12.Acne \n"
            "13.Bronchial Asthma 14.Hypertension 15.Migraine \n 16.Cervical spondylosis 17.Jaundice 18.Malaria \n"
            "19.Urinary tract infection 20.Allergy 21.Gastroesophageal reflux disease \n"
            "22.Drug reaction 23.Peptic ulcer disease 24.Diabetes"
        )
    st.button(label="Reset", on_click=refresh, type="primary")

user_input = st.text_input(label="text", placeholder="Enter your symptoms", label_visibility="hidden", key=120)
x = [user_input]
train.lower_case(x)
train.number_remove(x)
train.punctuation(x)
train.white_space(x)
train.token(x)
train.stopword(x)
train.pos_create(x)
train.convert(x)
train.pos_place(x)
train.lem(x)
Q = train.cv.transform(x)

logistic_probs = train.lr.predict_proba(Q)[0]
naive_probs = train.mnb.predict_proba(Q)[0]

logistic_pred = train.lr.classes_[np.argmax(logistic_probs)]
naive_pred = train.mnb.classes_[np.argmax(naive_probs)]

logistic_max_prob = np.max(logistic_probs)
naive_max_prob = np.max(naive_probs)

if logistic_max_prob > naive_max_prob:
    final_prediction = logistic_pred
    final_probability = logistic_max_prob
else:
    final_prediction = naive_pred
    final_probability = naive_max_prob

submit_button = st.button(label="Submit", type='primary')

if submit_button:
    if final_probability < 0.65:
        st.write("Need more info")
    else:
        st.write(f"You have been diagnosed with {final_prediction}")
        st.write(f"Probability: {final_probability:.2f}")
