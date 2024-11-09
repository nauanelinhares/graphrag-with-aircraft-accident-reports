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

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)

@st.cache_data
def loadSettings():
    llm = OpenAI(model="gpt-4o", api_key=OPEN_API_KEY, temperature=0.25)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding("dunzhang/stella_en_400M_v5", trust_remote_code=True)
    Settings.chunk_size = 512

def getLinks(query_str):
    llm = OpenAI(model="gpt-4o", api_key=OPEN_API_KEY, temperature=0)
    prompt = f"""
    Create a list of links based on the following query: {query_str}
    Example Output: https://www.example.com, https://www.example2.com
    
    I don't want other information, just the links.
    """
    resps= llm.complete(prompt=prompt)
    resps = resps.text.split(",")
    return resps

def readLinks(links):
    llm = OpenAI(model="gpt-4o", api_key=OPEN_API_KEY, temperature=0)
    docs = []
    for url in links:
        response = requests.get(url)
        response.raise_for_status()  # V
        tree = html.fromstring(response.content)
        cenipaLinks = tree.xpath('//a[contains(text(), "Clique aqui")]')
        if cenipaLinks:  
            for link in cenipaLinks:
                getPDFsLinksFromCenipa(link.get('href'))
        else:
            doc = SimpleWebPageReader(html_to_text=True).load_data([url])
            docs = docs + doc
    try:       
        doc = SimpleDirectoryReader("./Acidentes").load_data()
        docs = docs + doc
    except:
        pass
    
    index = createAnIndex(llm, docs)
    return index
                                                                    
            
def getPdfsFromCenipa(link, name):
    report_response = requests.get(link)
    report_response.raise_for_status()
    with open(f'Acidentes/report_{name}.pdf', 'wb') as file:
        file.write(report_response.content)
        print(f'Relatório {name} baixado com sucesso.')



def getPDFsLinksFromCenipa(cenipaLink):
    response = requests.get(cenipaLink)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

        # Parseando o conteúdo HTML da página
    tree = html.fromstring(response.content)
    links = tree.xpath('//a[@title="Relatório Final em Português"]')
    base_url = 'https://sistema.cenipa.fab.mil.br/cenipa/paginas/relatorios/'
    
    for _, link in enumerate(links, start=1):
        link_url =  link.get('href')
        report_url = base_url +link_url
        getPdfsFromCenipa(report_url, link_url)


    

def createAnIndex(llm, docs):
    loadSettings()
    index = PropertyGraphIndex.from_documents(
    docs,
    graph_store=graph_store,
    kg_extractor = SimpleLLMPathExtractor(llm=llm,
    max_paths_per_chunk=30,
    num_workers=6,
    ),
    include_embeddings=True,
    show_progress=True,
    Settings=Settings,
    ) 
    
    return index



@st.cache_data
def loadAdviceQueryEngine():
    df = pd.read_csv("./ocorrencias.csv")
    llm = OpenAI(model="gpt-4o", api_key=OPEN_API_KEY, temperature=0.25)
    instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. In general we will do like something df[df['History'].fillna('').str.contains('keyword', case=False)]"
    "3. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "4. The code should represent a solution to the query.\n"
    "5. PRINT ONLY THE EXPRESSION.\n"
    "6. Do not quote the expression.\n"
    )
    
    table_description = (
        """
        Descriptions of the columns:
         Occurrence Number: A unique identifier for each occurrence.
         Date: The date when the occurrence happened.
         Registration: The registration number of the aircraft involved in the occurrence.
         Classification: The classification of the occurrence (e.g., incident, serious incident, accident).
         Type: The type of occurrence (e.g., engine failure, controlled flight into terrain).
         Location: The location where the occurrence happened.
         State: The state where the occurrence happened.
         Aerodrome: The aerodrome associated with the occurrence.
         Operation: The type of operation being conducted at the time of the occurrence (e.g., private, agricultural).
         Status: The current status of the occurrence investigation (e.g., active, finalized).
         Link: A URL link to the detailed report of the occurrence.
         History: A brief description of the occurrence.
        """
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "That is the description of the dataframe:\n"
        "{table_description}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
    response_synthesis_prompt_str = (
        "Given an input question, show a response from the query results.\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: "
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5), table_description=table_description
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
    
    p = QueryPipeline(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
    p.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    p.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    # add link from response synthesis prompt to llm2
    p.add_link("response_synthesis_prompt", "llm2")
    return p


st.title('Conselhos de ocorrências aeronáuticas')

if "messages" not in st.session_state:
    st.session_state.messages = []

if "savedLinks" not in st.session_state:
    st.session_state.savedLinks = ''
    
if "index" not in st.session_state:
    st.session_state.index = ''
    
with st.spinner('Carregando banco de dados...'):
    p = loadAdviceQueryEngine()

if st.button("Limpar chat"):
    st.session_state.messages = []
    st.session_state.savedLinks = ''
    st.session_state.index = ''
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if question := st.chat_input("Digite sua pergunta aqui..."):
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner('Carregando resposta...'):
        message = st.chat_message("assistant")   
        response = p.run(query_str=f'{question}')
        message.write(response.message.content)
        st.session_state.messages.append({"role": "assistant", "content": response.message.content})
        
        try:
            links = getLinks(response.message.content)
        except:
            message.write("Tente novamente com outra pergunta.")
            
        if links != st.session_state.savedLinks and st.session_state.index == '':
            st.session_state.index = readLinks(links)
            st.session_state.savedLinks = links
            st.session_state.index.property_graph_store.save_networkx_graph(name="./kg.html")
            
        if st.session_state.index:
            
            query_engine = st.session_state.index.as_query_engine(include_text=True, llm=OpenAI(model="gpt-4o", api_key=OPEN_API_KEY, temperature=0.0))

            resp = query_engine.query(question)
            message.write(str(resp))
                
            st.session_state.messages.append({"role": "assistant", "content": str(resp)})





