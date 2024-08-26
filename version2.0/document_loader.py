from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import os
import glob

chunk_size = 500
chunk_overlap = 50
persist_directory = 'db'
embeddings_model_name = 'all-MiniLM-L6-v2'


loader_mapping = {'pdf': PyMuPDFLoader}
def document_loader(file_path, loader):
    loader_ = loader(file_path)
    return loader_.load()

def documents_loader(file_path):
    
    if type(file_path) == str:
        # print(file_path.split('.')[-1])
        loader = loader_mapping[file_path.split('.')[-1]]
        return document_loader(file_path, loader)
    elif type(file_path)== list:
        file_text = []
        for path in file_path:
            if path.split('.')[-1] in loader_mapping.keys():
                loader = loader_mapping[path.split('.')[-1]]
                file_text.extend(document_loader(path, loader))
        return file_text
    
def text_spliter(file_path):
    doc = documents_loader(file_path)
    if not doc:
        print('No Documents to Load')
    else:
        print(f'Loaded {len(doc)} new documents from File Uploader')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    
        texts = text_splitter.split_documents(doc)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
        return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        # Create and store locally vectorstore
    print("Creating new vectorstore")
    texts = text_spliter(file_path)
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")



file_path = 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf'
# file_path =[ 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf', 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf']
# print(text_spliter(file_path))
# print()

if __name__ == "__main__":
    main()