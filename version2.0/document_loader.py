from langchain_community.document_loaders import PyMuPDFLoader

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






file_path =[ 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf', 'D:\Prakash\Github\LLM\Teaching-Tutorial-Hydrogen-Bond.pdf']
print(documents_loader(file_path))
print()