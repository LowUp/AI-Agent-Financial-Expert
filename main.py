import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

load_dotenv()

LLM_API = os.getenv("LLM_API")

# Vectorise file data
def get_vectorstore(data):
    # DB_FAISS_PATH = "vectorestore/db_faiss"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    # db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    db = FAISS.from_documents(text_chunks, embeddings)
    # db.save_local(DB_FAISS_PATH)
    return db
    # return db
    # return len(text_chunks)

# Create conversation history
def get_conversation_chain(vecrorestore):
    llm = Ollama(
                model="llama3",
                base_url=LLM_API
            )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vecrorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain
           
def main():
    st.set_page_config(page_title="AI finance advisor", page_icon=":chart_with_upwards_trend:")
    st.title("AI finance advisor")
    try:
        
        user_csv = st.file_uploader("Upload your financial data", type="csv")
        
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        
        if user_csv is not None:
            
            # Save csv file locally
            with open(user_csv.name, "wb") as f:
                f.write(user_csv.getbuffer())
                
            # Process csv file
            with st.spinner("Processing..."):
                loader = CSVLoader(user_csv.name, encoding="utf-8")
                data = loader.load()
                
                vecrorestore = get_vectorstore(data)
                st.session_state.conversation = get_conversation_chain(vecrorestore)
                
                user_question = st.text_input("Enter a promt")
            
            # Let user enter a prompt and AI to response
            if user_question is not None and user_question != "":
                st.write(f"User promt: {user_question}")

                with st.spinner("Processing..."):
                    response = st.session_state.conversation({"question": user_question})
                    st.session_state.chat_history = response['chat_history']
                    
                    for i, message in enumerate(st.session_state.chat_history):
                        if i % 2 != 0:
                            st.write(f"AI response: {message.content}")
                
    except Exception as e:
        st.error(f"Error: {e}")
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()