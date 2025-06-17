# Q&A PDF Chatbot (Streamlit + Amazon Bedrock + TiDB)

This app allows users to upload a PDF document, generate Q&A pairs and text embeddings using Amazon Bedrock, store those embeddings in TiDB Cloud Serverless, and interact with the content via a chatbot interface.

---

## Features

- Upload and parse PDF files
- Create Q&A pairs from uploaded doc
- Chunk and embed document using Amazon Bedrock
- Store embeddings in TiDB Cloud Serverless
- Retrieve similar chunks using vector search (vector distance)
- Answer user queries 

---

## Tech Stack

- [Streamlit](https://streamlit.io/) – UI
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) – Embeddings + LLM
- [TiDB Serverless](https://tidb.cloud/) – Vector storage
- [LangChain](https://www.langchain.com/) – Text chunking and integration
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF parsing

---

## Deployment

- Clone this repo.
- Download root CA from https://letsencrypt.org/certs/isrgrootx1.pem and put it in `ssl` directory.
- Deploy on Streamlit Cloud. Go to https://streamlit.io/cloud.
- Click “New app” → Select your GitHub repo.
- Choose main branch and app.py as the main file.
- In Streamlit Cloud, go to your app → Settings → Secrets and add this following:

```toml
TIDB_HOST = "your_host"
TIDB_USER = "your_user"
TIDB_PASSWORD = "your_password"
TIDB_DATABASE = "your_database"
TIDB_SSL_CA = "/ssl/isrgrootx1.pem"

AWS_ACCESS_KEY_ID = "your_aws_key"
AWS_SECRET_ACCESS_KEY = "your_aws_secret"
AWS_REGION = "us-east-1"
```
---
