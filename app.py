import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(
    page_title="Diabetes Prediction Chatbot",
    page_icon="🩺"
)

st.title("Diabetes Prediction Chatbot")

@st.cache_resource
def load_model():
    with open('models/diabetes_model.pkl', 'rb') as f:
        data =pickle.load(f)
    return data['model'], data['scaler'], data['features']

model,scaler,features = load_model()

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Diabetes Risk Assessment Assistant. Let me check your risk factors."},
        {"role": "assistant", "content": "How many times have you been pregnant? (Enter 0 if not applicable)"}
    ]
    st.session_state.current_question = 0
    st.session_state.answers = {}

questions = [
    ("Pregnancies", "How many times have you been pregnant? (Enter 0 if not applicable)"),
    ("Glucose", "What is your plasma glucose concentration (mg/dL) from 2 hours in an oral glucose tolerance test?"),
    ("BloodPressure", "What is your diastolic blood pressure (mm Hg)?"),
    ("SkinThickness", "What is your triceps skin fold thickness (mm)?"),
    ("Insulin", "What is your 2-Hour serum insulin (mu U/ml)?"),
    ("BMI", "What is your body mass index (weight in kg/(height in m)^2)?"),
    ("DiabetesPedigreeFunction", "What is your diabetes pedigree function (a genetic risk score)?"),
    ("Age", "How old are you?")
]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt :=st.chat_input("Type your answer here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Store the answer
        current_feature = questions[st.session_state.current_question][0]
        st.session_state.answers[current_feature] = float(prompt)
        
        # Ask next question or show results
        if st.session_state.current_question < len(questions) - 1:
            st.session_state.current_question += 1
            next_question = questions[st.session_state.current_question][1]
            st.session_state.messages.append({"role": "assistant", "content": next_question})
        else:
            # All questions answered - make prediction
            input_df = pd.DataFrame([st.session_state.answers], columns=features)
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]
            
            if prediction == 1:
                result = f"🚨 Based on your inputs, there's a {probability*100:.1f}% chance you may have diabetes. Please consult a healthcare professional."
            else:
                result = f"✅ Based on your inputs, there's a {probability*100:.1f}% chance you don't have diabetes. Keep maintaining a healthy lifestyle!"
            
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.messages.append({"role": "assistant", "content": "Type 'restart' if you want to do another assessment."})
    
    except ValueError:
        st.session_state.messages.append({"role": "assistant", "content": "Please enter a valid number"})
    
    # Rerun to update the chat
    st.rerun()

# Handle restart
if st.session_state.messages and st.session_state.messages[-1]["content"].lower() == "restart":
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Diabetes Risk Assessment Assistant. Let's check your risk factors."},
        {"role": "assistant", "content": "How many times have you been pregnant? (Enter 0 if not applicable)"}
    ]
    st.session_state.current_question =0
    st.session_state.answers ={}
    st.rerun()