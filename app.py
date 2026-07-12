import streamlit as st
import joblib
import numpy as np

model = joblib.load("Models/logistic_model.pkl")
tfidf = joblib.load("Models/tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 Customer Support Ticket Classifier")
st.write("Enter a customer support ticket below to predict its category.")

ticket = st.text_area(
    "Enter Ticket Description",
    height=200,
    placeholder="Example: My Outlook mailbox is not opening and I cannot send emails."
)

predict_button = st.button("Predict Category")  

if predict_button:

    if ticket.strip() == "":
        st.warning("Please enter a ticket description.")

    else:
        ticket_tfidf = tfidf.transform([ticket])

        prediction = model.predict(ticket_tfidf)

        st.success(f"Predicted Category: {prediction[0]}")