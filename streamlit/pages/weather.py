import time
import streamlit as st
import avwx

import os
import nest_asyncio
import pandas as pd
from llama_index.core import (
    Document,
    PromptTemplate,
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
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.cohere_rerank import CohereRerank
import matplotlib.pyplot as plt
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the COHERE_API_KEY from environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")


@st.cache_data
def loadQueryEngine():
    df = pd.read_csv("./vra_filtered.csv")
    llm = OpenAI(model="gpt-4o", api_key=OPEN_API_KEY)
    instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
    )
    
    table_description = (
        """
        Descriptions of the columns:

        - Airline ICAO Code: ICAO code of the airline.
        - Airline: Name of the airline.
        - Flight Number: Flight number.
        - Equipment Model: Model of the equipment (aircraft).
        - Number of Seats: Number of seats in the aircraft.
        - Origin Airport ICAO Code: ICAO code of the origin airport.
        - Scheduled Departure: Scheduled departure time.
        - Actual Departure: Actual departure time.
        - Destination Airport ICAO Code: ICAO code of the destination airport.
        - Scheduled Arrival: Scheduled arrival time.
        - Actual Arrival: Actual arrival time.
        - Flight Status: Flight status (e.g., Accomplished, Canceled).
        - Justification: Justification (if any) for the flight status.
        - Reference: Date/time reference.
        - Departure Status: Departure status (e.g., On Time, Early, Delayed).
        - Arrival Status: Arrival status (e.g., On Time, Early, Delayed).
        - Reference Time for Departure: Reference time for departure.
        - Reference Time for Arrival: Reference time for arrival.
        - Delay: Flight delay in minutes.
        - Flight Time: Flight time in minutes. 
        - Origin Wind Direction: Wind direction at the origin airport.
        - Origin Wind Speed: Wind speed at the origin airport.
        - Origin Temperature: Temperature at the origin airport.
        - Origin Dew Point: Dew point at the origin airport.
        - Origin Visibility: Visibility at the origin airport.
        - Origin Pressure: Atmospheric pressure at the origin airport.
        - Origin Clouds: Cloud conditions at the origin airport.
        - Origin Weather Codes: Weather codes at the origin airport.
        - Origin Flight Rules: Flight rules at the origin airport (e.g., VFR, IFR).
        - Destination Wind Direction: Wind direction at the destination airport.
        - Destination Wind Speed: Wind speed at the destination airport.
        - Destination Temperature: Temperature at the destination airport.
        - Destination Dew Point: Dew point at the destination airport.
        - Destination Visibility: Visibility at the destination airport.
        - Destination Pressure: Atmospheric pressure at the destination airport.
        - Destination Clouds: Cloud conditions at the destination airport.
        - Destination Weather Codes: Weather codes at the destination airport.
        - Destination Flight Rules: Flight rules at the destination airport (e.g., VFR, IFR).
        - Origin Runway Threshold Hold: Hold time at the origin runway threshold.
        - Origin Time: Reference time at the origin airport.
        - Destination Runway Threshold Hold: Hold time at the destination runway threshold.
        - Destination Time: Reference time at the destination airport.
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
        "Given an input question, synthesize a response from the query results.\n"
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


st.title('Condições do tempo')


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner('Carregando banco de dados...'):
    p = loadQueryEngine()



selected = st.selectbox('De qual aeroporto você quer tirar sua dúvida?', ['SBKP', 'SBGR', 'SBBR', 'SBCF', 'SBSP', 'SBRJ', 'SBFL', 'SBCT',
       'SBRF', 'SBPA', 'SBSV', 'SBGL'])



with st.spinner('Carregando dados METAR...'):
    metar = avwx.Metar(selected)
    metar.update()
    time.sleep(2)
    
st.success("Dados METAR carregados com sucesso!")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Estação", value=metar.station.gps)
    st.metric(label="Direção do vento", value=f"{metar.data.wind_direction.value}°")
    st.metric(label="Temperatura", value=f"{metar.data.temperature.value}°C")

with col2:
    st.metric(label="Hora", value=metar.data.time.dt.strftime("%d/%m/%Y %H:%M"))
    st.metric(label="Velocidade do vento", value=f"{metar.data.wind_speed.value} nós")
    st.metric(label="Ponto de orvalho", value=f"{metar.data.dewpoint.value}°C")
    

with col3:
    st.metric(label="Visibilidade", value=f"{metar.data.visibility.value} metros")
    st.metric(label="Pressão", value=f"{metar.data.pressure_altitude} hPa")
    st.metric(label="Condição do voo", value=metar.data.flight_rules)
    
st.markdown(f"**Nuvens:** {str(metar.translations.clouds)}")
st.markdown(f"**Fenômenos meteorológicos:** {str(metar.translations.wx_codes)}")

if st.button("Limpar chat"):
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if question := st.chat_input("Digite sua pergunta aqui..."):
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner('Carregando resposta...'):
        message = st.chat_message("assistant")   
        response = p.run(query_str=question)
        message.write(response.message.content)
        st.session_state.messages.append({"role": "assistant", "content": response.message.content})



