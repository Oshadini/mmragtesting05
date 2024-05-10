import os
tesseract_path = "/usr/bin/tesseract"  # Replace with the actual path to tesseract
os.environ["PATH"] += os.pathsep + tesseract_path
from uuid import uuid4
import openai
import streamlit as st
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tempfile import NamedTemporaryFile
import tempfile
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#from langchain_google_vertexai import ChatVertexAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf
import uuid
import base64
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage
import uuid
from langchain.embeddings import VertexAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import io
import re
from IPython.display import HTML, display
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from PIL import Image
from langchain.chat_models import ChatOpenAI




st.set_page_config(layout='wide', initial_sidebar_state='expanded')


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi_model_rag_mvr"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
api_key = st.secrets["GOOGLE_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Multi-Modal RAG App`PDF`')

st.sidebar.subheader('Text Summarization Model')
time_hist_color = st.sidebar.selectbox('Summarize by', ('gpt-4-turbo', 'gemini-1.5-pro-latest'))

st.sidebar.subheader('Image Summarization Model')
immage_sum_model = st.sidebar.selectbox('Summarize by', ('gpt-4-vision-preview', 'gemini-1.5-pro-latest'))

#st.sidebar.subheader('Embedding Model')
#embedding_model = st.sidebar.selectbox('Select data', ('OpenAIEmbeddings', 'GoogleGenerativeAIEmbeddings'))

st.sidebar.subheader('Response Generation Model')
generation_model = st.sidebar.selectbox('Select data', ('gpt-4-vision-preview', 'gemini-1.5-pro-latest'))

#st.sidebar.subheader('Line chart parameters')
#plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
max_concurrecy = st.sidebar.slider('Maximum Concurrency', 3, 4, 5)

st.sidebar.markdown('''
---
Multi-Modal RAG App with Multi Vector Retriever
''')

uploaded_file = st.file_uploader(label = "Upload your file",type="pdf")
if uploaded_file is not None:
    temp_file="./temp.pdf"
    with open(temp_file,"wb") as file:
        file.write(uploaded_file.getvalue())

    image_path = "./"
    pdf_elements = partition_pdf(
        temp_file,
        chunking_strategy="by_title",
        #chunking_strategy="basic",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_path
    )

    # Categorize elements by type
    def categorize_elements(_raw_pdf_elements):
      """
      Categorize extracted elements from a PDF into tables and texts.
      raw_pdf_elements: List of unstructured.documents.elements
      """
      tables = []
      texts = []
      for element in _raw_pdf_elements:
          if "unstructured.documents.elements.Table" in str(type(element)):
              tables.append(str(element))
          elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
              texts.append(str(element))
      return texts, tables
    
    texts, tables = categorize_elements(pdf_elements)

    def encode_image(image_path):
        """Getting the base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    
    def image_summarize(img_base64, prompt):
        """Make image summary"""
        #model = ChatGoogleGenerativeAI(model="gemini-pro-vision", max_output_tokens=1024)
        #model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", max_output_tokens=1024)
    
        model = ChatOpenAI(
            temperature=0, model="gpt-4-vision-preview", openai_api_key = openai.api_key, max_tokens=1024)
    
        msg = model(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg.content
    
    def generate_img_summaries(path):
        """
        Generate summaries and base64 encoded strings for images
        path: Path to list of .jpg files extracted by Unstructured
        """
    
        # Store base64 encoded images
        img_base64_list = []
    
        # Store image summaries
        image_summaries = []
    
        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""
    
        # Apply to images
        for img_file in sorted(os.listdir(path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(image_summarize(base64_image, prompt))
    
        return img_base64_list, image_summaries
    
    fpath = "./figures"
    # Image summaries
    img_base64_list, image_summaries = generate_img_summaries(fpath)
    st.write(image_summaries)

