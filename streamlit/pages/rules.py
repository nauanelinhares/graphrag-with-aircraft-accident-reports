import streamlit as st
import requests
import os
import nest_asyncio
import pandas as pd
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import (
    Document,
    PromptTemplate,
    PropertyGraphIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_pipeline import InputComponent, Link, QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.postprocessor.cohere_rerank import CohereRerank
import matplotlib.pyplot as plt
import random
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from dotenv import load_dotenv
import nest_asyncio
from lxml import html
from llama_index.readers.web import SimpleWebPageReader



# Load environment variables from .env file
load_dotenv()

nest_asyncio.apply()

# Get the COHERE_API_KEY from environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
NEO4J_API_KEY = os.getenv("NEO4J_API_KEY")

llm = OpenAI(model="gpt-4o", api_key=OPEN_API_KEY, temperature=0.25)

def loadSettings():

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding("mixedbread-ai/mxbai-embed-2d-large-v1", trust_remote_code=True)
    Settings.chunk_size = 512

    
@st.cache_resource
def createAnIndexFromReportRules():
    loadSettings()
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context, index_id="vector_index")
    return index


def advancedRAG(index, llm):
    # use HyDE to hallucinate answer.
    retriever = index.as_retriever()
    reranker = CohereRerank(api_key=COHERE_API_KEY)
    summarizer = TreeSummarize(llm=llm)

    p = QueryPipeline(
        verbose=True
    )

    p.add_modules(
        {
            "llm": llm,
            "retriever": retriever,
            "summarizer": summarizer,
            "reranker": reranker,
        }
    )



    p.add_link("llm", "retriever")
    p.add_link("retriever", "reranker", dest_key="nodes")
    p.add_link("llm", "reranker", dest_key="query_str")
    p.add_link("reranker", "summarizer", dest_key="nodes")
    p.add_link("llm", "summarizer", dest_key="query_str")
    
    return p



st.title('Normas aeronáuticas')

with st.spinner('Carregando banco de dados...'):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "index" not in st.session_state:
        st.session_state.index = createAnIndexFromReportRules()
    

if st.button("Limpar chat"):
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if question := st.chat_input("Digite sua pergunta aqui..."):
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner('Carregando resposta...'):
        message = st.chat_message("assistant")   
        # p = advancedRAG(st.session_state.index, llm)
        # resp = p.run(topic=
        #              f"""
        #              {question}
        #              """
        #              )
        
        query_engine = st.session_state.index.as_query_engine()
        
        resp = query_engine.query(question)
        
        # print(response)
        
        message.write(str(resp))
        st.session_state.messages.append({"role": "assistant", "content": str(resp)})




