import streamlit as st

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

model_name = 'llama3.1:latest'
parser = StrOutputParser()
import random
import time

st.set_page_config(page_title = "Custom PDF ChatBot",
                   page_icon = "https://cdn.emojidex.com/emoji/seal/youtube.png",
                   layout = "wide",
                   initial_sidebar_state = "expanded",
                   menu_items = None)

st.cache_resource()
class llm_model:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = Ollama(model = model_name)
        self.embeddings = OllamaEmbeddings(model = model_name)
        self.parser = StrOutputParser()
        self.page_data = self.pdf_loader()
        self.vectorstore = self.vector_store()
        self.retriever = self.vectorstore.as_retriever()
        self.prompt = self.prompts()
        self.chain = self.chains()

    def pdf_loader(self):
        loader = PyPDFLoader(self.file_path)
        return loader.load_and_split()
    
    def vector_store(self):
        return DocArrayInMemorySearch.from_documents(self.page_data,
                                                    embedding = self.embeddings)
    
    def prompts(self):
        template = """
                    Answer the question based on the context below. If you can't 
                    answer the question, reply "I don't know".

                    Context: {context}

                    Question: {question}
                    """

        return ChatPromptTemplate.from_template(template)
    
    def chains(self):
        chain = (
                {
                    'context': itemgetter('question') | self.retriever, 
                    'question': itemgetter('question')
                } 
                |self.prompt
                |self.model
                |self.parser
                )
        return chain

    


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)





st.title(":blue[Custom PDF ChatBot Using Llama LLM (RAG)]")#📡

col1, col2 = st.columns([1,1], gap = 'medium')

with col1:
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )
    if file_upload is not None:
        file_path = file_upload.name
        st.write(file_upload.name)
        

        llm = llm_model(file_upload.name)
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
                        if llm: #st.session_state["vector_db"] is not None:
                            # response = process_question(
                            #     prompt, st.session_state["vector_db"], selected_model
                            # )
                            response = llm.chain.invoke({'question':prompt})
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.warning("Please upload a PDF file first.")
