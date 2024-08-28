import streamlit as st
import os
import glob

from chatbot import QA_Retrieval_LLM, parse_arguments



st.set_page_config(page_title = "Custom PDF ChatBot",
                   page_icon = "https://cdn.emojidex.com/emoji/seal/youtube.png",
                   layout = "wide",
                   initial_sidebar_state = "expanded",
                   menu_items = None)


    
st.cache_resource()
def model():
    return QA_Retrieval_LLM()
llm = model()
args = parse_arguments()





st.title(":blue[Custom PDF ChatBot Using Llama LLM (RAG)]")#📡

col1, col2 = st.columns([1,1], gap = 'medium')

with col1:
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=True
    )
    if file_upload is not None:
        # file_path = file_upload.names
        st.write([file_upload[i].name for i in range(len(file_upload))])
        # st.write(find_file_path(file_upload[0].name))
        

        # llm = llm_model(file_upload.name)
    else:
        st.warning("Please upload a PDF file first.")


with col2:

    message_container = st.container(height=500, border=True)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with message_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter Your Prompt here..."):
        # Display user message in chat message container
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        message_container.chat_message("user").markdown(prompt)

        response = f"User: {prompt}"
        # Display assistant response in chat message container
        with message_container.chat_message("assistant"):
                    with st.spinner(":blue[processing...]"):
                        if llm: 
                            response = llm(prompt)
                            answer, docs = response['result'], [] if args.hide_source else response['source_documents']

                            st.markdown(answer)
                            st.markdown(docs)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.warning("Please upload a PDF file first.")

