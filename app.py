import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import pickle
import joblib

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

llm = ChatGroq(model = 'gemma2-9b-it', api_key=groq_api_key)

st.set_page_config(page_title='Agriculture Yield Advisor', page_icon='🌾')
st.title('KrishiMitra AI')

if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

if "predicted_yield" not in st.session_state:
    st.session_state.predicted_yield = None

if "llm_explanation" not in st.session_state:
    st.session_state.llm_explanation = None
    
pipeline = joblib.load("crop_yield.joblib")
    
df = pd.read_csv('new_df.csv')
crop_list = sorted(df['Crop'].dropna().unique())
state_list = sorted(df['State'].dropna().unique())
season_list = sorted(df['Season'].dropna().unique())

crop = st.selectbox("Select Crop", crop_list, key="crop")
season = st.selectbox("Select Season", season_list, key="season")
state = st.selectbox("Select State", state_list, key="state")
st.session_state.area = st.number_input("Enter Cultivated Area (in hectares)")
st.session_state.rainfall = st.number_input("Enter Annual Rainfall Received (in mm)")
st.session_state.fertilizer = st.number_input("Enter Fertilizer Used (in kg)")
st.session_state.pesticide = st.number_input("Enter Pesticides Used (in kg)")

def predict_yield():
    input_data = pd.DataFrame(
        {
            'Season': [season], 'State' : [state], 'Annual_Rainfall': [st.session_state.rainfall], 'Fertilizer': [st.session_state.fertilizer], 'Pesticide': [st.session_state.pesticide], 'Crop': [crop], 'Area': [st.session_state.area]
        }
    )
    
    return pipeline.predict(input_data)[0]

if st.button('Predict'):
    
    pred = predict_yield()
    st.session_state.yield_pred = pred
    st.session_state.predicted_yield = pred

    
    prompt = f"""
    You are an agriculture expert.
    The predicted yield for {st.session_state.crop} on {st.session_state.area} hectares in {st.session_state.state}
    with {st.session_state.rainfall} mm rainfall, using {st.session_state.fertilizer} kg fertilizer and
    {st.session_state.pesticide} kg pesticide is {st.session_state.predicted_yield:.2f} tonnes.

    1. Explain why this yield is at this level based on the inputs.
    2. Suggest actionable steps to improve the yield.
    3. List relevant Indian government schemes for {st.session_state.crop} farmers in {st.session_state.state}.
    4. Keep the tone friendly and conversational for a farmer.
    After explaining everything, invite the farmer to ask follow-up questions.
    """

    try:
        response = llm.invoke(prompt)
        st.session_state.llm_explanation = response.content
       
        st.session_state.history.add_ai_message(response.content)
    except Exception as e:
        st.error(f"Failed to get explanation from LLM: {e}")

if st.session_state.llm_explanation:
    st.markdown(f"**Advisor : ** {st.session_state.llm_explanation}")
    
user_input = st.text_input("General Q&A related to the Prediction: ")
if st.button("Go") and user_input:
    st.session_state.history.add_user_message(user_input)
    chat_history_text = "\n".join(
        [f"User: {m.content}" if m.type == "human" else f"AI: {m.content}"
         for m in st.session_state.history.messages]
    )
    answer = llm.invoke(chat_history_text).content
    st.session_state.history.add_ai_message(answer)

    st.write(answer)
