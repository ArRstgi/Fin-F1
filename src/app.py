import streamlit as st
from src.rag_implementation import get_response

# set initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "Fin F1", "content": "Hi, I am the Fin F1 bot. What would you like to know about the Formula 1 Power Unit Financial Regulations ?"}
    ]
    
if "messages" in st.session_state.keys():
    # display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            
# get user input
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
        
        
if user_prompt is not None and st.session_state.messages[-1]["role"] != "Fin F1":
    with st.chat_message("Fin F1"):
        with st.spinner("Loading..."):
            rag_response = get_response(user_prompt)
            st.write(rag_response)
            
    new_rag_response = {"role": "Fin F1", "content": rag_response}
    st.session_state.messages.append(new_rag_response)