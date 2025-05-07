import streamlit as st
from transformers import pipeline

# Define function to load the translation pipeline
@st.cache_resource
def load_translation_pipeline(model_name):
    """
    Load the Hugging Face translation pipeline for the specified model.
    Device is set to CPU explicitly for compatibility.
    """
    return pipeline("translation", model=model_name, device=-1)  # Use CPU

# Language model mapping for translation
language_model_map = {
    ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
    ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
    ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es",
    ("English", "Italian"): "Helsinki-NLP/opus-mt-en-it",
    ("English", "Dutch"): "Helsinki-NLP/opus-mt-en-nl",
}

# Streamlit app
st.title("AI-Powered Translator")
st.write("Translate text between languages using Hugging Face's pre-trained models.")

# Input: Text to translate
text_to_translate = st.text_area("Enter text to translate:", placeholder="Type your text here...")

# Input: Language selection
source_language = st.selectbox("Source Language", options=["English"], index=0)
target_language = st.selectbox("Target Language", options=["German", "French", "Spanish", "Italian", "Dutch"], index=0)

# Translate button
if st.button("Translate"):
    if text_to_translate.strip():
        # Get the appropriate model
        model_name = language_model_map.get((source_language, target_language))
        if model_name:
            try:
                # Load translation pipeline and perform translation
                translator = load_translation_pipeline(model_name)
                translation = translator(text_to_translate)
                translated_text = translation[0]["translation_text"]
                st.success("Translation completed!")
                st.text_area("Translated Text", value=translated_text, height=150)
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
        else:
            st.error(f"Translation from {source_language} to {target_language} is not supported.")
    else:
        st.warning("Please enter text to translate.")

# Footer
st.markdown("---")
st.write("Powered by [Hugging Face Transformers](https://huggingface.co/transformers) and Streamlit.")
