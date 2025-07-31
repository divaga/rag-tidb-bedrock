import streamlit as st
import requests
import pymupdf
import os
import hashlib
import numpy as np
from typing import List, Dict, Any
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
import time
import io

# Configure page
st.set_page_config(
    page_title="TiDB + Bedrock RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'documents_count' not in st.session_state:
    st.session_state.documents_count = 0
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

class RAGSystem:
    def __init__(self):
        self.bedrock_client = None
        self.db_connection = None
        self.embedding_model = "amazon.titan-embed-text-v2:0"
        self.chat_model = "amazon.nova-micro-v1:0"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def initialize_bedrock(self, bearer_token: str, region_name: str = "us-east-1"):
        """Initialize Bedrock client with Bearer token"""
        try:
            if not bearer_token or not bearer_token.strip():
                st.error("Bedrock Bearer token is empty")
                return False
            
            # Store the bearer token for use in requests
            self.bearer_token = bearer_token
            self.region = region_name
            self.bedrock_endpoint = f"https://bedrock-runtime.{region_name}.amazonaws.com"
            
            # Test the connection by trying to invoke a simple model
            import requests
            
            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'Content-Type': 'application/json'
            }
            
            test_body = json.dumps({
                "inputText": "test"
            })
            
            test_url = f"{self.bedrock_endpoint}/model/{self.embedding_model}/invoke"
            
            response = requests.post(
                test_url,
                headers=headers,
                data=test_body,
                timeout=30
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                st.error("Bedrock Bearer token is invalid or expired")
                return False
            elif response.status_code == 403:
                st.error("Bearer token does not have permission to access Bedrock")
                return False
            else:
                st.error(f"Bedrock connection test failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error connecting to Bedrock: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Bedrock initialization failed: {str(e)}")
            return False
    
    def initialize_database(self, host: str, port: int, user: str, password: str, database: str):
        """Initialize TiDB connection and create table if needed"""
        try:
            self.db_connection = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                ssl_disabled=False,
                autocommit=True
            )
            
            # Create table if it doesn't exist
            cursor = self.db_connection.cursor()
            create_table_query = """
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_filename (filename),
                INDEX idx_file_hash (file_hash)
            )
            """
            cursor.execute(create_table_query)
            cursor.close()
            return True
        except Error as e:
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"PDF extraction failed: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from text file"""
        try:
            # Reset file pointer to beginning
            txt_file.seek(0)
            text = txt_file.read().decode('utf-8')
            return text
        except Exception as e:
            st.error(f"Text extraction failed: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Amazon Titan with Bearer token"""
        try:
            if not hasattr(self, 'bearer_token'):
                st.error("Bearer token not initialized")
                return []
            
            import requests
            
            # Prepare the request body for Titan embedding model
            body = json.dumps({
                "inputText": text
            })
            
            headers = {
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.bedrock_endpoint}/model/{self.embedding_model}/invoke"
            
            # Make the request
            response = requests.post(
                url,
                headers=headers,
                data=body,
                timeout=30
            )
            
            if response.status_code == 200:
                response_body = response.json()
                embedding = response_body.get('embedding', [])
                return embedding
            elif response.status_code == 429:
                st.error("Bedrock API rate limit exceeded. Please try again later.")
                return []
            elif response.status_code == 401:
                st.error("Bearer token is invalid or expired")
                return []
            elif response.status_code == 400:
                st.error("Invalid request to Bedrock embedding model")
                return []
            else:
                st.error(f"Bedrock embedding error: {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error during embedding generation: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")
            return []
    
    def check_document_exists(self, filename: str, file_hash: str) -> bool:
        """Check if document already exists in database"""
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT COUNT(*) FROM documents WHERE filename = %s AND file_hash = %s"
            cursor.execute(query, (filename, file_hash))
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except Error as e:
            st.error(f"Database query failed: {str(e)}")
            return False
    
    def store_document_chunks(self, filename: str, chunks: List[str], file_hash: str):
        """Store document chunks and embeddings in database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Delete existing chunks for this file (if re-uploading)
            delete_query = "DELETE FROM documents WHERE filename = %s"
            cursor.execute(delete_query, (filename,))
            
            # Insert new chunks
            insert_query = """
            INSERT INTO documents (filename, chunk_index, content, embedding, file_hash)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            for i, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk)
                if embedding:
                    embedding_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
                    cursor.execute(insert_query, (filename, i, chunk, embedding_str, file_hash))
            
            cursor.close()
            return True
        except Error as e:
            st.error(f"Database insertion failed: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity in TiDB"""
        try:
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []

            # Format the embedding as a TiDB VECTOR literal
            embedding_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"

            cursor = self.db_connection.cursor()

            # Query with vec_cosine_distance and VECTOR syntax
            query_sql = "SELECT id,filename,content,vec_cosine_distance(embedding,'" + str(embedding_str) + "') AS similarity FROM documents ORDER BY similarity ASC LIMIT " + str(top_k)
           
            cursor.execute(query_sql)
            results = cursor.fetchall()
            cursor.close()

            return [
                {
                    "id": row[0],
                    "filename": row[1],
                    "content": row[2],
                    "similarity": row[3]
                }
                for row in results
            ]
        except Error as e:
            st.error(f"Search failed: {str(e)}")
            return []
   
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Amazon Nova with Bearer token"""
        try:
            if not hasattr(self, 'bearer_token'):
                return "Bearer token not initialized"
            
            import requests
                
            # Prepare context
            context = "\n\n".join([chunk["content"] for chunk in context_chunks])
            
            # Create prompt for Nova
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Prepare the request body for Nova
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 1000,
                    "temperature": 0.7
                }
            })
            
            headers = {
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.bedrock_endpoint}/model/{self.chat_model}/invoke"
            
            # Make the request
            response = requests.post(
                url,
                headers=headers,
                data=body,
                timeout=60
            )
            
            if response.status_code == 200:
                response_body = response.json()
                answer = response_body['output']['message']['content'][0]['text']
                return answer
            elif response.status_code == 429:
                return "Rate limit exceeded: Please try again later"
            elif response.status_code == 401:
                return "Bearer token is invalid or expired"
            elif response.status_code == 400:
                return "Invalid request format"
            else:
                return f"Bedrock error: {response.status_code} - {response.text}"
            
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            st.error(f"Answer generation failed: {str(e)}")
            return "Sorry, I couldn't generate an answer at this time."
    
    def get_document_count(self) -> int:
        """Get total number of documents in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Error as e:
            return 0

def main():
    st.title("📚 RAG Document Q&A System (Bedrock Bearer Auth)")
    st.markdown("Upload documents and ask questions to get AI-powered answers using Amazon Bedrock with Bearer token authentication!")
    
    # Auto-initialize system on first run
    if not st.session_state.db_initialized:
        with st.spinner("🚀 Initializing system..."):
            try:
                # Get credentials from secrets
                bearer_token = st.secrets["BEDROCK_BEARER_TOKEN"]
                aws_region = st.secrets.get("AWS_REGION", "us-east-1")
                db_host = st.secrets["TIDB_HOST"]
                db_port = int(st.secrets["TIDB_PORT"])
                db_user = st.secrets["TIDB_USER"]
                db_password = st.secrets["TIDB_PASSWORD"]
                db_name = st.secrets["TIDB_DATABASE"]
                
                # Initialize RAG system
                rag = RAGSystem()
                
                # Initialize Bedrock
                if rag.initialize_bedrock(bearer_token, aws_region):
                    # Initialize database
                    if rag.initialize_database(db_host, db_port, db_user, db_password, db_name):
                        st.session_state.db_initialized = True
                        st.session_state.rag_system = rag
                        st.session_state.documents_count = rag.get_document_count()
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Database connection failed")
                        return
                else:
                    st.error("❌ Bedrock initialization failed")
                    return
                    
            except KeyError as e:
                st.error(f"❌ Missing secret: {e}")
                st.info("Please ensure all required secrets are configured in your Streamlit app:")
                st.code("""
                BEDROCK_BEARER_TOKEN = "your-bedrock-bearer-token"
                AWS_REGION = "us-east-1"  # optional, defaults to us-east-1
                TIDB_HOST = "your-tidb-host"
                TIDB_PORT = "4000"
                TIDB_USER = "your-username"
                TIDB_PASSWORD = "your-password"
                TIDB_DATABASE = "your-database-name"
                
                Note: Ensure your Bearer token has access to Amazon Nova Pro and Titan embeddings
                """)
                return
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                return
    
    # Main interface (only shown after initialization)
    rag = st.session_state.rag_system
    
    # Status bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", st.session_state.documents_count)
    with col2:
        st.metric("🔧 Status", "Ready" if st.session_state.db_initialized else "Not Ready")
    with col3:
        if st.session_state.last_query_time:
            st.metric("⏰ Last Query", st.session_state.last_query_time.strftime("%H:%M:%S"))
        else:
            st.metric("⏰ Last Query", "Never")
    
    # Reset button (optional - for development/debugging)
    with st.expander("🔧 System Controls", expanded=False):
        if st.button("🔄 Reset System", help="Reset system and reinitialize"):
            st.session_state.db_initialized = False
            st.session_state.rag_system = None
            st.session_state.documents_count = 0
            st.session_state.last_query_time = None
            st.rerun()
    
    st.divider()
    
    # Main functionality
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt'],
            help="Upload PDF or text files to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    text = rag.extract_text_from_pdf(uploaded_file)
                else:
                    text = rag.extract_text_from_txt(uploaded_file)
                
                if text:
                    # Generate file hash
                    file_hash = rag.get_text_hash(text)
                    
                    # Check if document already exists
                    if rag.check_document_exists(uploaded_file.name, file_hash):
                        st.warning("⚠️ Document already exists in the database!")
                    else:
                        # Chunk the text
                        chunks = rag.chunk_text(text)
                        
                        # Store chunks and embeddings
                        if rag.store_document_chunks(uploaded_file.name, chunks, file_hash):
                            st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                            st.info(f"📊 Created {len(chunks)} chunks")
                            st.session_state.documents_count = rag.get_document_count()
                        else:
                            st.error("❌ Failed to store document")
                else:
                    st.error("❌ Failed to extract text from document")
    
    with col2:
        st.header("❓ Ask a Question")
        query = st.text_area(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            height=100
        )
        
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if query:
                with st.spinner("🔍 Searching for relevant information..."):
                    # Search for similar chunks
                    similar_chunks = rag.search_similar_chunks(query, top_k=5)
                    
                    if similar_chunks:
                        with st.spinner("🤖 Generating answer..."):
                            # Generate answer
                            answer = rag.generate_answer(query, similar_chunks)
                            
                            # Display answer
                            st.subheader("💡 Answer:")
                            st.markdown(answer)
                            
                            # Display sources
                            st.subheader("📚 Sources:")
                            for i, chunk in enumerate(similar_chunks):
                                similarity_score = chunk['similarity']
                                color = "🟢" if similarity_score > 0.8 else "🟡" if similarity_score > 0.6 else "🔴"
                                
                                with st.expander(f"{color} Source {i+1}: {chunk['filename']} (Similarity: {similarity_score:.4f})"):
                                    st.write(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
                            
                            st.session_state.last_query_time = datetime.now()
                    else:
                        st.warning("⚠️ No relevant information found in the uploaded documents.")
            else:
                st.error("❌ Please enter a question.")

if __name__ == "__main__":
    main()