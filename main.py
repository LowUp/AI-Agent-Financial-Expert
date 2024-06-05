import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_csv_agent


def get_conversation_chain(llm, vectorstore):
    pass


def main1():
    st.set_page_config(page_title="AI Agent Code Generator", page_icon=":robot:")
    st.title("AI finance advisor agent")
    # st.header("AI Agent Code Generator")
    st.text_input("Enter your prompt")
    # st.button("send prompt")
    
    # with st.expander("AI Agent Code Generator"):
    #     st.write("AI Agent Code Generator")

    with st.sidebar:
        # st.subheader("AI Agent Code Generator")
        # st.write("AI Agent Code Generator")
        csv_docs =  st.file_uploader("Upload your csv file", accept_multiple_files=False, type=["csv"])
        if st.button("Process CSV"):
            with st.spinner("Processing..."):
                raw_data = csv_docs[0].read()
                st.write(raw_data)
                # st.write("Processing...")
                
def main():
    st.set_page_config(page_title="AI Agent Code Generator", page_icon=":robot:")
    st.title("AI finance advisor agent")
    
    user_csv = st.file_uploader("Upload your csv file", type="csv")
    
    if user_csv is not None:
        user_question = st.text_input("Enter a promt")
        
        llm = Ollama(
            model="llama3",
            base_url='http://192.168.1.65:11434'
        )
        agent = create_csv_agent(llm, user_csv, verbose=True, handle_parsing_errors=True)
        
        if user_question is not None and user_question != "":
            st.write(f"User promt: {user_question}")
            st.spinner("Processing...")
            response = agent.run(user_question)
            st.write(f"AI response: {response}")
        
        # with st.spinner("Processing..."):
        # if user_question is not None:
        #     st.spinner("Processing...")
        #     response = agent.run(user_question)
        #     st.write(response)
    

if __name__ == "__main__":
    main()