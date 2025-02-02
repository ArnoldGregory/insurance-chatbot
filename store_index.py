import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPinecone
from dotenv import load_dotenv
import pinecone
from src.helper import load_pdf, text_split, download_hugging_face_embeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Load API key from .env
PINECONE_ENV = "us-east-1"  # Your Pinecone environment (adjust if needed)
INDEX_NAME = "insurance"

# Initialize Pinecone using the new class-based approach
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load and process data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Extract text content
texts = [doc.page_content for doc in text_chunks]

# Generate embeddings
embedded_texts = embeddings.embed_documents(texts)

# Format data for Pinecone
vectors = [
    {
        "id": str(i),  
        "values": embedded_texts[i],  
        "metadata": {"text": texts[i]}  
    }
    for i in range(len(texts))
]

# Batch upsert (upload vectors in batches of 1000)
batch_size = 1000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i : i + batch_size]
    index.upsert(vectors=batch, namespace="insurance")

print("✅ Data upserted successfully!")
