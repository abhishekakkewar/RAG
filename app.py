import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings To generate Embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Specify your region here
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion from an uploaded PDF
def data_ingestion(uploaded_file):
    # Save the uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    
    # Clean up the temporary file
    os.remove(uploaded_file.name)
    
    temporary_docs = [doc.page_content for doc in docs]  # Only for preview purposes
    return docs, temporary_docs

# Vector Embedding and Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    # Create the Claude Model
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2_llm():
    # Create the Llama2 Model
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but summarize with at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            docs, temporary_docs = data_ingestion(uploaded_file)
            st.write("Preview of the extracted content (first 500 characters):")
            st.write(temporary_docs[0][:500])  # Show first 500 characters of the first document chunk for preview

            if st.button("Update Vectors"):
                with st.spinner("Creating vector store..."):
                    get_vector_store(docs)
                    st.success("Vector store updated!")

        user_question = st.text_input("Ask a Question from the PDF Files")
        
        if st.button("Claude Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_claude_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

        if st.button("Llama2 Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
                llm = get_llama2_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

if __name__ == "__main__":
    main()
