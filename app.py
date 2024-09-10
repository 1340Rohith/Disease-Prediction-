import os
import streamlit as st
import train
import pyautogui
os.environ['DISPLAY'] = ':0'
st.title("Disease Diagnosis")
def refresh():
    pyautogui.hotkey("ctrl","F5")

with st.sidebar:
    with st.expander("Identifiable Diseases"):
        st.write("1.Psoriasis  2.Varicose Veins  3.Typhoid \n 4.Chicken pox 5.Impetigo 6.Dengue \n 7.Fungal infection 8.Common Cold 9.Pneumonia \n 10.Dimorphic Hemorrhoids 11.Arthritis 12.Acne \n 13.Bronchial Asthma 14.Hypertension 15.Migraine \n 16.Cervical spondylosis 17.Jaundice 18.Malaria \n 19.urinary tract infection 20.allergy 21.gastroesophageal reflux disease \n 22.drug reaction 23.peptic ulcer disease, 24.diabetes")
    st.button(label="Reset",on_click=refresh,type="primary")
user_input = st.text_input(label="text",placeholder="Enter your symptoms",label_visibility="hidden",key=120)
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
logistic = train.lr.predict(Q)
naive = train.mnb.predict(Q)
submit_button = st.button(label="Submit", type = 'primary')
if submit_button:
    output1 = logistic
    output2 = logistic
    st.write(f"you have been diagnosed with {output1} (Logistic regression)")
    st.write(f"you have been diagnosed with {output2} (Naive Bayes)")
