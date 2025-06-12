import streamlit as st
import tempfile
import os
import boto3
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Bedrock
import mysql.connector
import numpy as np

# --- TiDB connection using Streamlit secrets ---
TIDB_CONFIG = {
    'host': st.secrets["TIDB_HOST"],
    'user': st.secrets["TIDB_USER"],
    'password': st.secrets["TIDB_PASSWORD"],
    'database': st.secrets["TIDB_DATABASE"],
    'ssl_ca': st.secrets["TIDB_SSL_CA"]
}

# --- Initialize Amazon Bedrock Clients using Streamlit secrets ---
boto3_bedrock = boto3.client(
    'bedrock-runtime',
    region_name=st.secrets["AWS_REGION"],
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
)

embedding_model = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-text-v1")
llm_model = Bedrock(client=boto3_bedrock, model_id="amazon.titan-text-lite-v1")

# --- PDF parsing and embedding ---
def parse_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def store_embeddings_in_tidb(text_chunks, embeddings):
    conn = mysql.connector.connect(**TIDB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_embeddings (
            id VARCHAR(64) PRIMARY KEY,
            chunk TEXT,
            embedding VECTOR(1536)
        )""")
    for chunk, emb in zip(text_chunks, embeddings):
        chunk_id = str(uuid.uuid4())
        emb_list = list(map(float, emb))
        cursor.execute("INSERT INTO pdf_embeddings (id, chunk, embedding) VALUES (%s, %s, %s)",
                       (chunk_id, chunk, emb_list))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_similar_chunks(query, k=3):
    conn = mysql.connector.connect(**TIDB_CONFIG)
    cursor = conn.cursor()
    query_emb = embedding_model.embed_query(query)
    cursor.execute("SELECT chunk, embedding FROM pdf_embeddings")
    chunks = []
    for chunk, emb in cursor.fetchall():
        emb = np.array(emb, dtype=float)
        score = np.linalg.norm(np.array(query_emb) - emb)
        chunks.append((score, chunk))
    cursor.close()
    conn.close()
    chunks.sort()
    return [chunk for _, chunk in chunks[:k]]

def generate_answer(context, question):
    prompt = f"""
    You are a helpful assistant. Based on the following context, answer the question:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return llm_model(prompt)

# --- Streamlit UI ---
st.title("📄 PDF-based RAG Chatbot with Amazon Bedrock and TiDB")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    text = parse_pdf(uploaded_file)
    chunks = split_text(text)
    st.info(f"Parsed and split into {len(chunks)} text chunks.")
    embeddings = embedding_model.embed_documents(chunks)
    store_embeddings_in_tidb(chunks, embeddings)
    st.success("Embeddings stored in TiDB.")

query = st.text_input("Ask a question based on uploaded content:")
if query:
    similar_chunks = fetch_similar_chunks(query)
    context = "\n".join(similar_chunks)
    response = generate_answer(context, query)
    st.write("### Answer:")
    st.write(response)

st.markdown("---")
st.markdown("Built with 🧠 Amazon Bedrock, 🐬 TiDB Serverless, and 📚 LangChain")
