from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size = 500
chunk_overlap = 50

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
    
def text_spliter(filepath):
    doc = documents_loader(file_path)
    if not doc:
        print('No Documents to Load')
    else:
        print(f'Loaded {len(doc)} new documents from File Uploader')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    
        texts = text_splitter.split_documents(doc)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
        return texts




file_path = 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf'
file_path =[ 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf', 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf']
print(text_spliter(file_path))
print()