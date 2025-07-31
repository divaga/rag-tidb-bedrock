# Q&A PDF Chatbot (Streamlit + Amazon Bedrock + TiDB)

This app allows users to upload a PDF document, generate Q&A pairs and text embeddings using Amazon Bedrock, store those embeddings in TiDB Cloud Serverless, and interact with the content via a chatbot interface.

---

## Features

- Upload and parse PDF files
- Chunk and embed document using Amazon Bedrock
- Store embeddings in TiDB Cloud Serverless
- Retrieve similar chunks using vector search (vector distance)
- Answer user queries 

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
TIDB_PORT = "your_database_port"
TIDB_SSL_CA = "/ssl/isrgrootx1.pem"

AWS_BEARER_TOKEN_BEDROCK = "your_aws_bedrock_key"

```
---
