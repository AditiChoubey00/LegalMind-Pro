# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.chains.summarize import load_summarize_chain
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
# import torch
# import base64

# #MODEL AND TOKENIZER

# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype= torch.float32)

# #file Loader and preprocessing

# def file_preprocessing(file):
#     loader=PyPDFLoader(file)
#     pages =loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts= text_splitter.split_documents(pages)
#     final_texts = ""
#     for text in texts:
#         print(text)
#         final_texts = final_texts + text.page_content
#     return final_texts

# # LM Pipeline

# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer = tokenizer,
#         max_length = 2000,
#         min_length = 200
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

# @st.cache_data
# #function to display the PDF of a given file
# def displayPDF(file):
#     #Opening file from file path
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
   
#     #Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64, {base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

#     #Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # streamlit code
# st.set_page_config(layout='wide', page_title="Summarization App")

# def main():

#     st.title('Document Summarization App using Language Model')

#     uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

#     if uploaded_file is not None:
#         if st.button("Summarize"):
#             col1, col2=st.columns(2)
#             filepath="data/"+uploaded_file.name
#             with open(filepath,'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())
#             with col1:
#                 st.info("Uploaded PDF File")
#                 pdf_viewer=displayPDF(filepath)
#                 # pdf_path=uploaded_file.name
#                 # displayPDF(pdf_path)
            
#             with col2:
#                 st.info("Summarization is below")
#                 # summary=llm_pipeline(pdf_path)
#                 # st.success(summary)

#                 summary=llm_pipeline(filepath)
#                 st.success(summary)

# if __name__=='__main__':
#     main()




#WORKING 
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
# import torch
# import base64

# # MODEL AND TOKENIZER
# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# # FILE LOADER AND PREPROCESSING
# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = " ".join(text.page_content for text in texts)
#     return final_texts

# # LLM PIPELINE
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=2000,
#         min_length=500
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     summary = result[0]['summary_text']
    
#     # Ensure line breaks for better readability
#     summary = summary.replace(". ", ".\n")
    
#     return summary

# @st.cache_data
# def displayPDF(file):
#     """Function to display the PDF file in Streamlit."""
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # STREAMLIT APP
# st.set_page_config(layout='wide', page_title="Summarization App")

# def main():
#     st.title('Document Summarization App using Language Model')

#     uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

#     if uploaded_file is not None:
#         if st.button("Summarize"):
#             col1, col2 = st.columns(2)
#             filepath = f"data/{uploaded_file.name}"

#             with open(filepath, 'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())

#             with col1:
#                 st.info("Uploaded PDF File")
#                 displayPDF(filepath)

#             with col2:
#                 st.info("Summarization is below")
#                 summary = llm_pipeline(filepath)

#                 # Use `st.text_area()` for better word wrapping
#                 st.text_area("Summary", summary, height=300)

# if __name__ == '__main__':
#     main()













#WORKING WITH EXPORT WORD N PDF
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
# import torch
# import base64
# from docx import Document  # For Word document creation
# from fpdf import FPDF  # For PDF creation

# # MODEL AND TOKENIZER
# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# # FILE LOADER AND PREPROCESSING
# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = " ".join(text.page_content for text in texts)
#     return final_texts

# # LLM PIPELINE
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=2000,
#         min_length=500
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     summary = result[0]['summary_text']
    
#     # Ensure line breaks for better readability
#     summary = summary.replace(". ", ".\n")
    
#     return summary

# @st.cache_data
# def displayPDF(file):
#     """Function to display the PDF file in Streamlit."""
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # EXPORT FUNCTIONS
# def export_to_word(summary, filename):
#     """Export summary to a Word document."""
#     doc = Document()
#     doc.add_paragraph(summary)
#     doc.save(filename)

# def export_to_pdf(summary, filename):
#     """Export summary to a PDF file."""
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, summary)
#     pdf.output(filename)

# # STREAMLIT APP
# st.set_page_config(layout='wide', page_title="Summarization App")

# def main():
#     st.title('Document Summarization App using Language Model')
#     uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])
    
#     if uploaded_file is not None:
#         if st.button("Summarize"):
#             col1, col2 = st.columns(2)
#             filepath = f"data/{uploaded_file.name}"
            
#             # Save the uploaded file locally
#             with open(filepath, 'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())
            
#             with col1:
#                 st.info("Uploaded PDF File")
#                 displayPDF(filepath)
            
#             with col2:
#                 st.info("Summarization is below")
#                 summary = llm_pipeline(filepath)
                
#                 # Display the summary
#                 st.text_area("Summary", summary, height=300)
                
#                 # Export options
#                 st.subheader("Export Summary")
#                 word_filename = "summary.docx"
#                 pdf_filename = "summary.pdf"
                
#                 # Export to Word
#                 export_to_word(summary, word_filename)
#                 with open(word_filename, "rb") as word_file:
#                     st.download_button(
#                         label="Download as Word (.docx)",
#                         data=word_file,
#                         file_name=word_filename,
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                     )
                
#                 # Export to PDF
#                 export_to_pdf(summary, pdf_filename)
#                 with open(pdf_filename, "rb") as pdf_file:
#                     st.download_button(
#                         label="Download as PDF (.pdf)",
#                         data=pdf_file,
#                         file_name=pdf_filename,
#                         mime="application/pdf"
#                     )

# if __name__ == '__main__':
#     main()





# # MULTILINGUAL 
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
# import torch
# import base64
# from docx import Document  # For Word document creation
# from fpdf import FPDF  # For PDF creation

# # MODEL AND TOKENIZER FOR SUMMARIZATION
# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# # FILE LOADER AND PREPROCESSING
# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = " ".join(text.page_content for text in texts)
#     return final_texts

# # LLM PIPELINE FOR SUMMARIZATION
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=2000,
#         min_length=500
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     summary = result[0]['summary_text']
    
#     # Ensure line breaks for better readability
#     summary = summary.replace(". ", ".\n")
    
#     return summary

# # TRANSLATION PIPELINE
# def translate_text(text, target_language):
#     """Translate text to the target language."""
#     translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}")
#     translated_text = translator(text, max_length=2000)[0]['translation_text']
#     return translated_text

# @st.cache_data
# def displayPDF(file):
#     """Function to display the PDF file in Streamlit."""
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # EXPORT FUNCTIONS
# def export_to_word(summary, filename):
#     """Export summary to a Word document."""
#     doc = Document()
#     doc.add_paragraph(summary)
#     doc.save(filename)

# def export_to_pdf(summary, filename):
#     """Export summary to a PDF file."""
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, summary)
#     pdf.output(filename)

# # STREAMLIT APP
# st.set_page_config(layout='wide', page_title="Multilingual Summarization App")

# def main():
#     st.title('Multilingual Document Summarization App')

#     # Language options
#     language_options = {
#         "English": "en",
#         "Hindi": "hi",
#         "German": "de",
#         "French": "fr",
#         "Spanish": "es",
#         "Chinese": "zh",
#     }

#     uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])
#     selected_language = st.selectbox("Select Output Language", list(language_options.keys()))

#     if uploaded_file is not None:
#         if st.button("Summarize"):
#             col1, col2 = st.columns(2)
#             filepath = f"data/{uploaded_file.name}"
            
#             # Save the uploaded file locally
#             with open(filepath, 'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())
            
#             with col1:
#                 st.info("Uploaded PDF File")
#                 displayPDF(filepath)
            
#             with col2:
#                 st.info("Summarization is below")
                
#                 # Generate English summary
#                 summary = llm_pipeline(filepath)
                
#                 # Translate summary to the selected language
#                 target_language_code = language_options[selected_language]
#                 if target_language_code != "en":
#                     summary = translate_text(summary, target_language_code)
                
#                 # Display the summary
#                 st.text_area("Summary", summary, height=300)
                
#                 # Export options
#                 st.subheader("Export Summary")
#                 word_filename = "summary.docx"
#                 pdf_filename = "summary.pdf"
                
#                 # Export to Word
#                 export_to_word(summary, word_filename)
#                 with open(word_filename, "rb") as word_file:
#                     st.download_button(
#                         label="Download as Word (.docx)",
#                         data=word_file,
#                         file_name=word_filename,
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                     )
                
#                 # Export to PDF
#                 export_to_pdf(summary, pdf_filename)
#                 with open(pdf_filename, "rb") as pdf_file:
#                     st.download_button(
#                         label="Download as PDF (.pdf)",
#                         data=pdf_file,
#                         file_name=pdf_filename,
#                         mime="application/pdf"
#                     )

# if __name__ == '__main__':
#     main()






#MULTILINGUAL 2
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
from docx import Document  # For Word document creation
from fpdf import FPDF  # For PDF creation

# MODEL AND TOKENIZER FOR SUMMARIZATION
checkpoint = "LaMini-Flan-T5-248M"
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

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# EXPORT FUNCTIONS
def export_to_word(summary, filename):
    doc = Document()
    doc.add_paragraph(summary)
    doc.save(filename)

def export_to_pdf(summary, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(filename)

# STREAMLIT APP
st.set_page_config(layout='wide', page_title="Multilingual Summarization App")

def main():
    st.title('Multilingual Document Summarization App')

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
                displayPDF(filepath)
            
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

if __name__ == '__main__':
    main()



# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
# import torch
# import base64
# from docx import Document  # For Word document creation
# from fpdf import FPDF  # For PDF creation
# import requests
# from bs4 import BeautifulSoup
# import re

# # MODEL AND TOKENIZER
# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# # FILE LOADER AND PREPROCESSING
# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = " ".join(text.page_content for text in texts)
#     return final_texts

# # LLM PIPELINE FOR SIMPLE SUMMARY
# def llm_pipeline_simple_summary(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=2000,
#         min_length=500
#     )
#     input_text = file_preprocessing(filepath)
    
#     # Simplify the prompt to encourage simpler language
#     prompt = f"Simplify the following text into plain language: {input_text}"
#     result = pipe_sum(prompt)
#     summary = result[0]['summary_text']
    
#     # Ensure line breaks for better readability
#     summary = summary.replace(". ", ".\n")
    
#     return summary

# # SCRAPE LEGAL TERM MEANINGS FROM GOOGLE
# def get_legal_term_meaning(term):
#     """Scrape the meaning of a legal term from Google."""
#     query = f"{term} meaning"
#     url = f"https://www.google.com/search?q={query}"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, 'html.parser')
#         # Look for the first definition in the search results
#         definition = soup.find("div", {"class": "BNeawe iBp4i AP7Wnd"})
#         if definition:
#             return definition.get_text(strip=True)
#     return None

# # EXTRACT LEGAL TERMS FROM THE DOCUMENT
# def extract_legal_terms(text):
#     """Extract legal terms from the document."""
#     # Assume legal terms are capitalized words longer than 3 characters
#     legal_terms = set(re.findall(r'\b[A-Z][A-Za-z]{3,}\b', text))
#     return legal_terms

# # DISPLAY PDF IN STREAMLIT
# @st.cache_data
# def displayPDF(file):
#     """Function to display the PDF file in Streamlit."""
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # EXPORT FUNCTIONS
# def export_to_word(summary, filename):
#     """Export summary to a Word document."""
#     doc = Document()
#     doc.add_paragraph(summary)
#     doc.save(filename)

# def export_to_pdf(summary, filename):
#     """Export summary to a PDF file."""
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, summary)
#     pdf.output(filename)

# # STREAMLIT APP
# st.set_page_config(layout='wide', page_title="Summarization App")

# def main():
#     st.title('Document Summarization App using Language Model')
#     uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])
    
#     if uploaded_file is not None:
#         if st.button("Summarize"):
#             col1, col2 = st.columns(2)
#             filepath = f"data/{uploaded_file.name}"
            
#             # Save the uploaded file locally
#             with open(filepath, 'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())
            
#             with col1:
#                 st.info("Uploaded PDF File")
#                 displayPDF(filepath)
            
#             with col2:
#                 st.info("Simplified Summary is below")
                
#                 # Generate a simple summary
#                 summary = llm_pipeline_simple_summary(filepath)
#                 st.text_area("Simplified Summary", summary, height=300)
                
#                 # Extract legal terms from the entire document
#                 full_text = file_preprocessing(filepath)
#                 legal_terms = extract_legal_terms(full_text)
                
#                 # Fetch meanings for legal terms
#                 term_definitions = {}
#                 for term in legal_terms:
#                     meaning = get_legal_term_meaning(term)
#                     if meaning:
#                         term_definitions[term] = meaning
                
#                 # Display the meaning section
#                 if term_definitions:
#                     st.subheader("Meaning of Legal Terms")
#                     for term, meaning in term_definitions.items():
#                         st.markdown(f"**{term}**: {meaning}")
#                 else:
#                     st.info("No legal terms found in the document.")
                
#                 # Export options
#                 st.subheader("Export Summary")
#                 word_filename = "simplified_summary.docx"
#                 pdf_filename = "simplified_summary.pdf"
                
#                 # Export to Word
#                 export_to_word(summary, word_filename)
#                 with open(word_filename, "rb") as word_file:
#                     st.download_button(
#                         label="Download as Word (.docx)",
#                         data=word_file,
#                         file_name=word_filename,
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                     )
                
#                 # Export to PDF
#                 export_to_pdf(summary, pdf_filename)
#                 with open(pdf_filename, "rb") as pdf_file:
#                     st.download_button(
#                         label="Download as PDF (.pdf)",
#                         data=pdf_file,
#                         file_name=pdf_filename,
#                         mime="application/pdf"
#                     )

# if __name__ == '__main__':
#     main()




# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
# import torch
# import base64
# from docx import Document  # For Word document creation
# from fpdf import FPDF  # For PDF creation
# import requests
# from bs4 import BeautifulSoup
# import re

# # MODEL AND TOKENIZER
# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# # FILE LOADER AND PREPROCESSING
# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = " ".join(text.page_content for text in texts)
#     return final_texts

# # LLM PIPELINE FOR SIMPLE SUMMARY
# def llm_pipeline_simple_summary(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=2000,
#         min_length=500
#     )
#     input_text = file_preprocessing(filepath)
    
#     # Simplify the prompt to encourage simpler language
#     prompt = f"Simplify the following text into plain language: {input_text}"
#     result = pipe_sum(prompt)
#     summary = result[0]['summary_text']
    
#     # Ensure line breaks for better readability
#     summary = summary.replace(". ", ".\n")
    
#     return summary

# # SCRAPE LEGAL TERM MEANINGS FROM GOOGLE
# def get_legal_term_meaning(term):
#     """Scrape the meaning of a legal term from Google."""
#     query = f"{term} meaning"
#     url = f"https://www.google.com/search?q={query}"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }
#     try:
#         response = requests.get(url, headers=headers)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.content, 'html.parser')
#             # Look for the first definition in the search results
#             definition = soup.find("div", {"class": "BNeawe iBp4i AP7Wnd"})
#             if definition:
#                 return definition.get_text(strip=True)
#     except Exception as e:
#         print(f"Error fetching meaning for '{term}': {e}")
#     return None

# # EXTRACT LEGAL TERMS FROM THE DOCUMENT
# def extract_legal_terms(text):
#     """Extract legal terms from the document."""
#     # Assume legal terms are capitalized words longer than 3 characters
#     legal_terms = set(re.findall(r'\b[A-Z][A-Za-z]{3,}\b', text))
#     return legal_terms

# # DISPLAY PDF IN STREAMLIT
# @st.cache_data
# def displayPDF(file):
#     """Function to display the PDF file in Streamlit."""
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # EXPORT FUNCTIONS
# def export_to_word(summary, filename):
#     """Export summary to a Word document."""
#     doc = Document()
#     doc.add_paragraph(summary)
#     doc.save(filename)

# def export_to_pdf(summary, filename):
#     """Export summary to a PDF file."""
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, summary)
#     pdf.output(filename)

# # STREAMLIT APP
# st.set_page_config(layout='wide', page_title="Summarization App")

# def main():
#     st.title('Document Summarization App using Language Model')
#     uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])
    
#     if uploaded_file is not None:
#         if st.button("Summarize"):
#             col1, col2 = st.columns(2)
#             filepath = f"data/{uploaded_file.name}"
            
#             # Save the uploaded file locally
#             with open(filepath, 'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())
            
#             with col1:
#                 st.info("Uploaded PDF File")
#                 displayPDF(filepath)
            
#             with col2:
#                 st.info("Simplified Summary is below")
                
#                 # Generate a simple summary
#                 summary = llm_pipeline_simple_summary(filepath)
#                 st.text_area("Simplified Summary", summary, height=300)
                
#                 # Extract legal terms from the entire document
#                 full_text = file_preprocessing(filepath)
#                 legal_terms = extract_legal_terms(full_text)
                
#                 # Fetch meanings for legal terms
#                 term_definitions = {}
#                 for term in legal_terms:
#                     meaning = get_legal_term_meaning(term)
#                     if meaning:
#                         term_definitions[term] = meaning
                
#                 # Display the meaning section
#                 if term_definitions:
#                     st.subheader("Meaning of Legal Terms")
#                     for term, meaning in term_definitions.items():
#                         st.markdown(f"**{term}**: {meaning}")
#                 else:
#                     st.info("No legal terms found in the document.")
                
#                 # Export options
#                 st.subheader("Export Summary")
#                 word_filename = "simplified_summary.docx"
#                 pdf_filename = "simplified_summary.pdf"
                
#                 # Export to Word
#                 export_to_word(summary, word_filename)
#                 with open(word_filename, "rb") as word_file:
#                     st.download_button(
#                         label="Download as Word (.docx)",
#                         data=word_file,
#                         file_name=word_filename,
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                     )
                
#                 # Export to PDF
#                 export_to_pdf(summary, pdf_filename)
#                 with open(pdf_filename, "rb") as pdf_file:
#                     st.download_button(
#                         label="Download as PDF (.pdf)",
#                         data=pdf_file,
#                         file_name=pdf_filename,
#                         mime="application/pdf"
#                     )

# if __name__ == '__main__':
#     main()