import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_objectbox import ObjectBox
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
google_api_key=os.getenv('GOOGLE_API_KEY')
st.title("CHAT WITH GCUH")

llm=ChatGoogleGenerativeAI(google_api_key=google_api_key,model="gemini-pro")

prompt=ChatPromptTemplate.from_template("""You are an excellent reader of the documents so give me the answer as according to the given context <context>{context}<context> for the question:{input}""")

def vectors_embeddings():
  if "vectors" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings()
    st.session_state.document=PyPDFLoader("GCUH.pdf")
    st.session_state.docs=st.session_state.document.load()
    st.session_state.text_split=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=200)
    st.session_state.docs_split=st.session_state.text_split.split_documents(st.session_state.docs[:10])
    st.session_state.vectors=ObjectBox.from_documents(st.session_state.docs_split,embedding=st.session_state.embeddings,embedding_dimensions=768)

query=st.text_input("Please Enter your Query")
if st.button("Document Embeddings"):
  vectors_embeddings()

if query:
  documnet_chain=create_stuff_documents_chain(llm,prompt)
  retreiver=st.session_state.vectors.as_retriever()
  retreval_chain=create_retrieval_chain(retreiver,documnet_chain)
  
  response=retreval_chain.invoke({"input":query})
  
  st.write(response['answer'])
