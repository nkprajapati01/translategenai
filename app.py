import streamlit as st
from transformers import pipeline

# Load Hugging Face translation pipeline
@st.cache_resource
def load_translation_pipeline():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")  # Default model: English to German

translator = load_translation_pipeline()

# App title
st.title("AI-Powered Translator")
st.write("Translate text between languages using Hugging Face's pre-trained models.")

# Input text
text_to_translate = st.text_area("Enter text to translate:", placeholder="Type your text here...")

# Language selection
source_language = st.selectbox("Source Language", options=["English"], index=0)
target_language = st.selectbox("Target Language", options=["German", "French", "Spanish", "Italian", "Dutch"], index=0)

# Translation models mapping
language_model_map = {
    ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
    ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
    ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es",
    ("English", "Italian"): "Helsinki-NLP/opus-mt-en-it",
    ("English", "Dutch"): "Helsinki-NLP/opus-mt-en-nl",
}

# Translate button
if st.button("Translate"):
    if text_to_translate.strip():
        # Load appropriate translation model based on selected languages
        model_name = language_model_map.get((source_language, target_language))
        if model_name:
            translator = pipeline("translation", model=model_name)
            translation = translator(text_to_translate)
            translated_text = translation[0]["translation_text"]
            st.success("Translation completed!")
            st.text_area("Translated Text", value=translated_text, height=150)
        else:
            st.error(f"Translation from {source_language} to {target_language} is not supported.")
    else:
        st.warning("Please enter text to translate.")

# Footer
st.markdown("---")
st.write("Powered by [Hugging Face Transformers](https://huggingface.co/transformers) and Streamlit.")
