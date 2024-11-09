import streamlit as st

# Assuming client is an instance of some class, you need to define it

# Set the page configuration
st.set_page_config(page_title='Main App', page_icon=':airplane:', layout='wide', initial_sidebar_state='auto')


# Common elements for all pages
st.sidebar.title('Barra de tarefas')
st.sidebar.write("""Este é uma tela para falar com chatbot que coleta informações sobre ocorrências aeronáuticas, normas e dados de voos passados. O chatbot foi criado para um trabalho de graduação 
                 em 2024""")

# Define the navigation menu
pg = st.navigation(
  [
   st.Page("pages/weather.py", title="Dados de Voos passados", icon="🌦️"),
      st.Page("pages/rules.py", title="Normas", icon="📃"),
   st.Page("pages/advice.py", title="Oocorrências Aeronáuticas", icon="🛬")
   ]
)

pg.run()


