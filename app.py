from dotenv import load_dotenv
import os, string
import google.generativeai as genai
import streamlit as st
import pandas as pd


def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing all characters
    - Removing punctuation
    - Stripping leading/trailing whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path

load_dotenv()

st.title("Hello, GeminiAI")
st.write("This is your first Streamlit app.")


# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found.  Please check the file path.")

with col2:
    if st.button("Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    st.subheader("Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader("Dataset Preview")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]

    st.dataframe(filtered_df)

    # Sentiment Graph
    st.subheader("Sentiment by Product")
    gourped = st.session_state["df"].groupby(["PRODUCT"])["SENTIMENT_SCORE"].mean()
    st.bar_chart(gourped)
