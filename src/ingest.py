import sys
sys.path.append("")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
    BSHTMLLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import torch
from utils import *



def create_vector_db(DATA_PATH: str = 'web_data/', 
                     DB_FAISS_PATH: str = 'vectorstores/db_faiss/', 
                     device = 'cpu'):
    documents = []
    for f in os.listdir(DATA_PATH):
        path = f'{DATA_PATH}/' + f
        try:
            if f.endswith(".pdf"):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())

            elif f.endswith(".html"):
                loader = BSHTMLLoader(path)
                documents.extend(loader.load())

            elif f.endswith(".docx"):
                loader = Docx2txtLoader(path)
                documents.extend(loader.load())

            elif f.endswith(".txt") or f.endswith(".md"):
                loader = TextLoader(path)
                documents.extend(loader.load())
            
            elif f.endswith(".ppt") or f.endswith(".pptx"):
                
                if f.endswith(".ppt"):
                    # Change the file name to "name.pptx"
                    new_path = os.path.join(os.path.dirname(path), os.path.splitext(f)[0] + ".pptx")

                    # Rename the file
                    os.rename(path, new_path)

                loader = UnstructuredPowerPointLoader(path)
                documents.extend(loader.load())

        except Exception as e:
            print("issue with ",f)
            print('Error:',e)
            pass


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': device})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"Done processing! {len(documents)} documents processed.") 


def main():
    DATA_PATH="web_data"
    DB_FAISS_PATH = "vectorstores/db_faiss/"
    device = take_device()
    create_vector_db(DATA_PATH, DB_FAISS_PATH, device)

if __name__=="__main__":
    main()