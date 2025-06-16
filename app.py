import streamlit as st
import tempfile
import os
import boto3
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
import mysql.connector
import numpy as np
import json

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
llm_model = Bedrock(client=boto3_bedrock, model_id="mistral.mistral-7b-instruct-v0:2")

# --- PDF parsing and Q&A generation ---
def parse_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def generate_qa_pairs(chunk):
    prompt = f"""
    Extract up to 3 question-answer pairs from the following text:

    {chunk}

    Respond in JSON format like:
    [{{"question": "...", "answer": "..."}}, ...]
    """
    response = llm_model(prompt)
    try:
        return json.loads(response)
    except:
        return []

def store_qa_embeddings_in_tidb(qa_pairs):
    conn = mysql.connector.connect(**TIDB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_embeddings (
            id VARCHAR(64) PRIMARY KEY,
            question TEXT,
            answer TEXT,
            embedding VECTOR(1536)
        )""")
    for qa in qa_pairs:
        question = qa.get("question")
        answer = qa.get("answer")
        if question and answer:
            embedding = embedding_model.embed_query(question)
            qa_id = str(uuid.uuid4())
            st.write("Embedding: ",str(embedding))
            #cursor.execute("INSERT INTO qa_embeddings (id, question, answer, embedding) VALUES (%s, %s, %s, %s)",
            #               (qa_id, question, answer, embedding_str))
    #conn.commit()
    #cursor.close()
    conn.close()

def fetch_best_answer(user_question, k=1):
    conn = mysql.connector.connect(**TIDB_CONFIG)
    cursor = conn.cursor()
    query_emb = embedding_model.embed_query(user_question)
    cursor.execute("SELECT a.question,a.answer,vec_cosine_distance(a.embedding,'" + query_emb + "') as score FROM qa_embeddings a ORDER BY score ASC LIMIT 3")
    candidates = []
    for question, answer, score in cursor.fetchall():
        candidates.append((score, question, answer))
    cursor.close()
    conn.close()
    candidates.sort()
    return candidates[:k]

# --- Streamlit UI ---
st.title("📄 PDF Q&A Chatbot with Amazon Bedrock and TiDB")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    text = parse_pdf(uploaded_file)
    chunks = split_text(text)
    all_qa_pairs = []
    for chunk in chunks:
        qa_pairs = generate_qa_pairs(chunk)
        all_qa_pairs.extend(qa_pairs)
    st.info(f"Generated {len(all_qa_pairs)} Q&A pairs.")
    store_qa_embeddings_in_tidb(all_qa_pairs)
    st.success("Q&A pairs embedded and stored in TiDB.")

query = st.text_input("Ask a question:")
if query:
    answers = fetch_best_answer(query)
    if answers:
        st.write("### Most relevant answer:")
        st.write(answers[0][2])  # Display the best answer
    else:
        st.warning("No matching answer found.")

st.markdown("---")
st.markdown("Built with 🧠 Amazon Bedrock, 🐬 TiDB Serverless, and 📚 LangChain")
