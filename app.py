import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import os
from docx import Document
from fpdf import FPDF
import asyncio
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

# Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)

# MODEL AND TOKENIZER FOR SUMMARIZATION
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# FILE LOADER AND PREPROCESSING
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = " ".join(text.page_content for text in texts)
    return final_texts

# LLM PIPELINE FOR SUMMARIZATION
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=2000,
        min_length=500
    )
    input_text = file_preprocessing(filepath)
    
    if not input_text.strip():
        raise ValueError("Input text is empty. Unable to generate a summary.")
    
    result = pipe_sum(input_text)
    summary = result[0]['summary_text']
    
    if not summary.strip():
        raise ValueError("Generated summary is empty. Please check the input document.")
    
    summary = summary.replace(". ", ".\n")
    return summary

# TRANSLATION PIPELINE
def translate_text(text, target_language):
    if not text.strip():
        raise ValueError("Input text for translation is empty.")
    
    # Truncate the text to fit within the model's token limit
    tokenizer = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}").tokenizer
    tokenized_text = tokenizer.tokenize(text)
    max_tokens = 512  # Adjust based on the model's limit
    truncated_text = tokenizer.convert_tokens_to_string(tokenized_text[:max_tokens])
    
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}")
    try:
        translated_text = translator(truncated_text, max_length=2000)[0]['translation_text']
    except IndexError as e:
        raise ValueError(f"Translation failed. Ensure the input text is valid and within the model's token limit. Error: {e}")
    
    return translated_text

# FASTAPI APP
app = FastAPI()

# Pydantic model for API input
class SummarizeRequest(BaseModel):
    filepath: str
    target_language: str = "en"  # Default to English

# API endpoint for summarization
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        # Generate summary
        summary = llm_pipeline(request.filepath)
        
        # Translate summary if needed
        if request.target_language != "en":
            summary = translate_text(summary, request.target_language)
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# STREAMLIT APP
def streamlit_app():
    st.set_page_config(layout='wide', page_title="Multilingual Summarization App")

    language_options = {
        "English": "en",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Chinese": "zh",
    }

    uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])
    selected_language = st.selectbox("Select Output Language", list(language_options.keys()))

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = f"data/{uploaded_file.name}"
            
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            
            with col1:
                st.info("Uploaded PDF File")
                # Extract and display PDF text
                reader = PdfReader(filepath)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text()
                st.text_area("PDF Content", pdf_text, height=300)
            
            with col2:
                st.info("Summarization is below")
                
                try:
                    summary = llm_pipeline(filepath)
                    st.write("English Summary:", summary)
                except ValueError as e:
                    st.error(f"Error generating summary: {e}")
                    return
                
                target_language_code = language_options[selected_language]
                if target_language_code != "en":
                    try:
                        summary = translate_text(summary, target_language_code)
                        st.write(f"Translated Summary ({selected_language}):", summary)
                    except ValueError as e:
                        st.error(f"Error translating summary: {e}")
                        return
                
                st.text_area("Summary", summary, height=300)
                
                st.subheader("Export Summary")
                word_filename = "summary.docx"
                pdf_filename = "summary.pdf"
                
                export_to_word(summary, word_filename)
                with open(word_filename, "rb") as word_file:
                    st.download_button(
                        label="Download as Word (.docx)",
                        data=word_file,
                        file_name=word_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                
                export_to_pdf(summary, pdf_filename)
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="Download as PDF (.pdf)",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )

# Run both Streamlit and FastAPI
if __name__ == '__main__':
    import threading

    # Run FastAPI in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    # Run Streamlit in the main thread
    def run_streamlit():
        streamlit_app()

    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # Run Streamlit
    run_streamlit()