import sys
sys.path.append("")
import os
import torch
import pickle
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup as Soup


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    RecursiveUrlLoader, 
    DirectoryLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    BSHTMLLoader,
    UnstructuredImageLoader,
    Docx2txtLoader,
    TextLoader,
    SeleniumURLLoader,
    YoutubeLoader,
    UnstructuredURLLoader,
    NewsURLLoader,
    UnstructuredPowerPointLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import TokenTextSplitter

from src.utils import *


class VectorDatabase:
    def __init__(self, 
                    data_path: str = 'data/web_data/', 
                    data_index_path: str = 'data/data_index.csv',
                     db_faiss_path: str = 'data/vectorstores/db_faiss/', 
                     model_config_path: str = 'config/model_config.yaml',
                     data_url_path: str = 'config/data_config.yaml',
                     device = 'cpu'):
        
        self.data_path = data_path
        self.db_faiss_path = db_faiss_path
        self.device = device
        self.data_index_path = data_index_path
        self.data_index = self._read_data_index()
        
        check_path('data')
        
        self.data_url = config_parser(data_config_path = data_url_path)
        self.model_config = config_parser(data_config_path = model_config_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_config.get('embedding_model'),
                                model_kwargs={'device': take_device()})
        
    def _read_data_index(self):
        if os.path.exists(self.data_index_path):
            # Load the existing data index
            data_index = pd.read_csv(self.data_index_path)
        else:
            # Create a new DataFrame if the file doesn't exist
            data_index = pd.DataFrame(columns=['file_path', 'extension', 'backup_date'])
        return data_index
    
    def is_file_in_db(self, file_path):
        return file_path in self.data_index['file_path'].values


    def read_url_data(self):
        documents = []
        file_path_list = []

        single_url_list = self.data_url.get('single_url_list')
        if single_url_list:
            for link in single_url_list:
                if not self.is_file_in_db(str(link)):
                    loader = NewsURLLoader(urls=[link], 
                                        post_processors=[clean_extra_whitespace],)
                    documents.extend(loader.load())
                    file_path_list.extend([link])


        youtube_url_list = self.data_url.get('youtube_url_list')
        if youtube_url_list:
            for link in youtube_url_list:
                if not self.is_file_in_db(str(link)): 
                    loader = YoutubeLoader.from_youtube_url(
                        link, add_video_info=False
                    )
                    documents.extend(loader.load())
                    file_path_list.append(link)
        
        recursive_url_list = self.data_url.get('recursive_url_list')
        if recursive_url_list :
            for link in recursive_url_list:
                if not self.is_file_in_db(str(link)):
                    loader = RecursiveUrlLoader(
                        url=link, max_depth=2,
                        extractor=lambda x: Soup(x, "html.parser").text,
                    )
                    documents.extend(loader.load())
                    file_path_list.append(link)
        return documents, file_path_list

    def read_file_data(self):
        documents = []
        file_path_list = []

        for f in os.listdir(self.data_path):
            path = f'{self.data_path}/' + f
            try:
                if f.endswith(".pdf") and not self.is_file_in_db(path):
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                    file_path_list.append(path)

                elif f.endswith(".html") and not self.is_file_in_db(path):
                    loader = BSHTMLLoader(path)
                    documents.extend(loader.load())
                    file_path_list.append(path)

                elif f.endswith(".docx") and not self.is_file_in_db(path):
                    loader = Docx2txtLoader(path)
                    documents.extend(loader.load())
                    file_path_list.append(path)

                elif (f.endswith(".txt") or f.endswith(".md")) and not self.is_file_in_db(path):
                    loader = TextLoader(path)
                    documents.extend(loader.load())
                    file_path_list.append(path)
                
                elif f.endswith(".ppt") or f.endswith(".pptx") and not self.is_file_in_db(path):
                    
                    if f.endswith(".ppt"):
                        # Change the file name to "name.pptx"
                        new_path = os.path.join(os.path.dirname(path), os.path.splitext(f)[0] + ".pptx")

                        # Rename the file
                        os.rename(path, new_path)
                    loader = UnstructuredPowerPointLoader(path)
                    documents.extend(loader.load())
                    file_path_list.append(path)

            except Exception as e:
                print("issue with ",f)
                print('Error:',e)
                pass
        return documents, file_path_list
    
    @timeit
    def read_data(self):
        documents1, file_path_list1 = self.read_file_data()
        documents2, file_path_list2 = self.read_url_data()
        documents = documents1 + documents2
        file_path_list = file_path_list1 + file_path_list2
        
        return documents, file_path_list

    
    def _split_documents(self, documents):
        text_splitter = TokenTextSplitter(chunk_size=self.model_config.get('splitter_chunk_size'),
                                           chunk_overlap=self.model_config.get('splitter_chunk_overlap'))
        texts = text_splitter.split_documents(documents)
        return texts
    
    def save_vector_database(self, texts):
        new_db = FAISS.from_documents(texts, self.embeddings)
        try:
            old_db = self.load_vector_db()
            old_db.merge_from(new_db)
            old_db.save_local(self.db_faiss_path)
        except:
            new_db.save_local(self.db_faiss_path)

        print(f"Vector database saved in {self.db_faiss_path}.")



    def save_data_index(self, file_path_list):
        data_index = self._read_data_index()
        current_date = datetime.now().strftime('%Y-%m-%d')
        file_info = [(str(path), os.path.splitext(os.path.basename(path))[1][1:], current_date) for path in file_path_list]
        new_data = pd.DataFrame(file_info, columns=['file_path', 'extension', 'backup_date'])

        # Append the new data to the existing data index
        data_index = pd.concat([data_index, new_data], ignore_index=True)
        data_index.to_csv(self.data_index_path, index=False)
        print (f'Data index saved to CSV in {self.data_index_path}')

    @timeit
    def create_vector_db(self):
        documents, file_path_list = self.read_data()
        if documents:
            texts = self._split_documents(documents)
            self.save_vector_database(texts)
            self.save_data_index(file_path_list)
            print(f"Done processing! {len(documents)} documents processed.")
        return documents, file_path_list
    
    @timeit
    def load_vector_db(self):
        try:
            database = FAISS.load_local(self.db_faiss_path, self.embeddings)
        except Exception as e:
            print ('Error:',e)
            return None
        return database
    

def main():
    vector_db = VectorDatabase()
    documents, file_path_list = vector_db.create_vector_db() 
    print('file_path_list', file_path_list)

if __name__=="__main__":
    main()