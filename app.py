import streamlit as st
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "linear_svc_cefr_model.joblib"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)


def run_app():
    st.set_page_config(
        page_title="CEFR English Predictor",
        page_icon="📚",
    )

    model = load_model()
    vectorizer = load_vectorizer()

    st.title("📚 English CEFR Predictor")

    st.write(
        "Enter an English writing sample to predict its CEFR level."
    )

    text = st.text_area(
        "English text",
        height=250,
        placeholder="Write or paste your English text here...",
    )

    if st.button("Predict CEFR Level", type="primary"):

        if not text.strip():
            st.warning("Please enter some text first.")
            return

        X = vectorizer.transform([text])

        prediction = model.predict(X)[0]

        st.divider()

        st.subheader("Predicted CEFR Level")

        st.success(f"## {prediction}")
