# from ollama import Client
# from llama_index.llms.ollama import Ollama
# from llama_parse import LlamaParse
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
# from llama_index.legacy.llms.ollama import Ollama
# from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Ollama


# client = Client(host='http://192.168.1.65:11434/')

# result = client.chat(model="llama3", messages=[{
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   }])

# print(result)

# llm = Ollama(model="mistral", reques_timeout=30, base_url='http://192.168.1.65:11434')

# result = llm.complete("Hello World")

# print(result)

# from llama_index.legacy.tools import QueryEngineTool, ToolMetadata


# from llama_index.legacy.llms.ollama import Ollama
# from llama_index.legacy import LlamaParse
# from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
# from llama_index.legacy.embeddings import resolve_embed_model
# from llama_index.legacy.tools import QueryEngineTool, ToolMetadata
# from llama_index.legacy.agent import ReActAgent
# from pydantic import BaseModel
# from llama_index.legacy.output_parsers import PydanticOutputParser
# from llama_index.legacy.core.query_pipeline import QueryPipeline
# from prompts import context, code_parser_template
# from code_reader import code_reader
# from dotenv import load_dotenv
# import os
# import ast

# load_dotenv(dotenv_path="env-variables/.env")

# llm = Ollama(model="mistral", request_timeout=30.0, base_url='http://192.168.1.65:11434')

# parser = LlamaParse(result_type="markdown")

# file_extractor = {".pdf": parser}
# documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# embed_model = resolve_embed_model("local:BAAI/bge-m3")
# vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# query_engine = vector_index.as_query_engine(llm=llm)

# result = query_engine.query("What are some of the routes in the api ?")

# print(result)

import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_csv_agent

llm = Ollama(model="mistral", 
            #  reques_timeout=30, 
             base_url='http://192.168.1.65:11434'
             )
query = "Tell me a joke"

for chunks in llm.stream(query):
    print(chunks)