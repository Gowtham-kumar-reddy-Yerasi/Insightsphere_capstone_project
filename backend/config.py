import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT","us-east-1") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index") 

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Embedding Model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths (adjust as needed)
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR", "data")

# Groq chat model (llama3-70b-8192 and llama-3.3-70b-versatile have been shut down)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
