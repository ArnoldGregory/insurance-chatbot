from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import sys

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
app = Flask(__name__)

load_dotenv()

from src.helper import download_hugging_face_embeddings


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key="pcsk_6KC7X7_C63yRFNLptAy8xikMD6eFsbXJRgDgtfUETSB4i3GPKZxLT1aSDxQDfGZYhd8iMr")
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


index_name="insurance"

#Loading the index
batch=Pinecone.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=batch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)