# 📄 RAG PDF Chatbot (Streamlit + Amazon Bedrock + TiDB)

This app allows users to upload a PDF document, generate text embeddings using Amazon Bedrock, store those embeddings in TiDB Cloud Serverless, and interact with the content via a chatbot interface.

---

## 🚀 Features

- Upload and parse PDF files
- Chunk and embed document using Amazon Titan Embedding model (`amazon.titan-embed-text-v1`)
- Store embeddings in TiDB Cloud Serverless
- Retrieve similar chunks using basic vector search
- Answer user queries using Claude v2 (`anthropic.claude-v2`) from Amazon Bedrock

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/) – UI
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) – Embeddings + LLM
- [TiDB Serverless](https://tidb.cloud/) – Vector storage
- [LangChain](https://www.langchain.com/) – Text chunking and integration
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF parsing

---

