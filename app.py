import streamlit as st
import joblib

# ==========================
# Load Model
# ==========================

model = joblib.load("Models/logistic_model.pkl")
tfidf = joblib.load("Models/tfidf_vectorizer.pkl")

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# ==========================
# Header
# ==========================

st.title("🎫 Customer Support Ticket Classifier")

st.write("""
Automatically classify IT support tickets into the appropriate support team
using **Natural Language Processing (NLP)** and **Machine Learning**.
""")

st.divider()

# ==========================
# About Project
# ==========================

st.subheader("📌 About this Project")

st.info("""
This project uses **TF-IDF Vectorization** along with a **Logistic Regression**
classifier to automatically predict the category of customer support tickets.
It was developed as an end-to-end NLP project to demonstrate text preprocessing,
feature engineering, model training, evaluation, and deployment using Streamlit.
""")

st.divider()

# ==========================
# Categories
# ==========================

st.subheader("🎯 Supported Categories")

col1, col2 = st.columns(2)

with col1:
    st.success("🔑 Access")
    st.success("💻 Hardware")
    st.success("📦 Storage")
    st.success("👨‍💼 HR Support")

with col2:
    st.success("🛒 Purchase")
    st.success("🔒 Administrative Rights")
    st.success("📁 Internal Project")
    st.success("📄 Miscellaneous")

st.divider()

# ==========================
# Model Information
# ==========================

st.subheader("📊 Model Information")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "85.8%")
c2.metric("F1 Score", "85.8%")
c3.metric("Dataset", "47,837")
c4.metric("Classes", "8")

st.write("""
**Algorithm :** Logistic Regression

**Vectorizer :** TF-IDF Vectorizer

**Train-Test Split :** 80 : 20
""")

st.divider()

# ==========================
# Prediction Section
# ==========================

st.subheader("📝 Enter Ticket Description")

ticket = st.text_area(
    "",
    height=180,
    placeholder="Example: My Outlook mailbox is full and I cannot send emails."
)

if st.button("🚀 Predict Category", use_container_width=True):

    if ticket.strip() == "":
        st.warning("Please enter a ticket description.")

    else:

        ticket_tfidf = tfidf.transform([ticket])

        prediction = model.predict(ticket_tfidf)

        st.success(f"### ✅ Predicted Category : **{prediction[0]}**")

st.divider()

# ==========================
# Sample Inputs
# ==========================

st.subheader("🧪 Try These Sample Tickets")

with st.expander("Click to view sample tickets"):

    st.markdown("""
**1.** My mailbox is full.

➡ Expected Category: **Storage**

---

**2.** My laptop keyboard is not working properly.

➡ Expected Category: **Hardware**

---

**3.** I am unable to install a software..

➡ Expected Category: **Administrative Rights**

---

**4.** I am unable to reset my password..

➡ Expected Category: **Access**

---

**5.** I need to purchase new items.

➡ Expected Category: **Purchase**

---

**6.** Please report on time.

➡ Expected Category: **HR Support**
""")

st.divider()

# ==========================
# Disclaimer
# ==========================

st.subheader("⚠️ Disclaimer")

st.warning("""
This project was developed **for learning and portfolio purposes only**.

The model is trained using a traditional Machine Learning approach
(Logistic Regression + TF-IDF). While it performs well on the evaluation
dataset (≈86% accuracy), predictions may occasionally be incorrect for
ambiguous, incomplete, or previously unseen ticket descriptions.
""")

st.divider()

# ==========================
# Footer
# ==========================

st.caption(
    "Developed by **Amandeep Singh** • B.Tech CSE • NLP & Machine Learning Project"
)