from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = "pcsk_6KC7X7_C63yRFNLptAy8xikMD6eFsbXJRgDgtfUETSB4i3GPKZxLT1aSDxQDfGZYhd8iMr"
# HOST_URL = "https://insurance-207ckai.svc.aped-4627-b74a.pinecone.io"
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key="pcsk_6KC7X7_C63yRFNLptAy8xikMD6eFsbXJRgDgtfUETSB4i3GPKZxLT1aSDxQDfGZYhd8iMr")
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


index_name="insurance"

#Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

# Format the embeddings correctly
vectors = [
    {
        "id": str(i),  # Unique string ID
        "values": embeddings[i],  # The actual vector
        "metadata": {"text": text_chunks[i]}  # Metadata (optional)
    }
    for i in range(len(text_chunks))
]

# ✅ Batch the vectors (split into chunks of 1000)
batch_size = 1000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i : i + batch_size]
    index.upsert(vectors=batch, namespace="insurance")