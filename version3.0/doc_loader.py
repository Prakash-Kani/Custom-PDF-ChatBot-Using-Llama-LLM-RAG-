from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

embeddings_model_name =  "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

from langchain.document_loaders import PyMuPDFLoader
file_path = r'F:\prakash\chathistory\chemicalbonding.pdf'

loader = PyMuPDFLoader(file_path)
pages = loader.load()

chunk_size = 1000
chunk_overlap = 200
persist_directory = 'physics-chemical'
embeddings_model_name = 'all-MiniLM-L6-v2'

text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
texts = text_splitter.split_documents(pages)
file_name = 'physics'
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, collection_name=file_name)


