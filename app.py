import os
import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize AWS Bedrock client and embeddings model
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Define prompt template
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but summarize with at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Data ingestion function to load and split documents
def data_ingestion(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create FAISS vector store from documents
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

# Function to load FAISS vector store
def load_vector_store():
    return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

# Function to get Claude LLM from AWS Bedrock
def get_claude_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

# Function to get Llama2 LLM from AWS Bedrock
def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Function to get response from LLM
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Main function to handle Streamlit UI and logic
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        st.write(f"Processing {uploaded_file.name}...")

        if st.button("Update Vectors"):
            with st.spinner("Updating vectors..."):
                docs = data_ingestion(uploaded_file)
                get_vector_store(docs)
                st.success("Vector store updated!")

        user_question = st.text_input("Ask a question based on the PDF content")

        if st.button("Claude Output"):
            with st.spinner("Generating Claude response..."):
                faiss_index = load_vector_store()
                llm = get_claude_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Done")

        if st.button("Llama2 Output"):
            with st.spinner("Generating Llama2 response..."):
                faiss_index = load_vector_store()
                llm = get_llama2_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Done")

if __name__ == "__main__":
    main()

