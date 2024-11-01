import streamlit as st

# Assuming client is an instance of some class, you need to define it

# Set the page configuration
st.set_page_config(page_title='Main App', page_icon=':airplane:', layout='wide', initial_sidebar_state='auto')


# Common elements for all pages
st.sidebar.title('Common Sidebar')
st.sidebar.write('This is a common sidebar element.')

# Define the navigation menu
pg = st.navigation(
  [
   st.Page("pages/config.py", title="Config", icon="🔥"),
   st.Page("pages/weather.py", title="Tempo", icon="🌦️"),
   st.Page("pages/advice.py", title="Conselhos", icon="🛬")
   ]
)

pg.run()


