import os
import streamlit as st
st.title('Streamlit Environment Debug')
for key in ['STREAMLIT_SERVER_RUNNING', 'STREAMLIT_RUN_MAIN', 'STREAMLIT_APP_NAME', 'STREAMLIT_RUN_ID', 'STREAMLIT_SERVER_PORT']:
    st.write(key, os.environ.get(key))
st.write('All env vars:', {k: v for k, v in os.environ.items() if 'STREAMLIT' in k})
