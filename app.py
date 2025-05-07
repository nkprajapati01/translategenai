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
st.title("🌐 LingoMate: Your AI-Powered Translator Buddy 🤖")
st.write("#### Effortless translations at your fingertips! 🌍✨")
st.markdown(
    """
    Welcome to **LingoMate**, your friendly AI-powered translator!  
    Break down language barriers and communicate seamlessly.  
    Simply enter your text, select the source and target languages, and let the magic happen! 🚀
    """
)

# Input: Text to translate
st.subheader("💬 Enter Text to Translate")
text_to_translate = st.text_area("Type your text below:", placeholder="E.g., Hello, how are you?")

# Input: Language selection
st.subheader("🌍 Select Languages")
source_language = st.selectbox("🌐 Source Language", options=["English"], index=0)
target_language = st.selectbox("🌐 Target Language", options=["German", "French", "Spanish", "Italian", "Dutch"], index=0)

# Translate button
if st.button("✨ Translate Now!"):
    if text_to_translate.strip():
        # Get the appropriate model
        model_name = language_model_map.get((source_language, target_language))
        if model_name:
            try:
                # Load translation pipeline and perform translation
                translator = load_translation_pipeline(model_name)
                translation = translator(text_to_translate)
                translated_text = translation[0]["translation_text"]
                st.success("✅ Translation completed! Here's your result:")
                st.text_area("🎯 Translated Text", value=translated_text, height=150)
            except Exception as e:
                st.error(f"⚠️ An error occurred during translation: {e}")
        else:
            st.error(f"⚠️ Translation from {source_language} to {target_language} is not supported.")
    else:
        st.warning("⚠️ Please enter some text to translate.")

# Footer
st.markdown("---")
st.write("🌟 **Powered by [Hugging Face Transformers](https://huggingface.co/transformers) and Streamlit**. Breaking language barriers, one word at a time! 🌟")
