import sys
sys.path.append("")

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

from src.utils import *
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from src.ingest import VectorDatabase
from langchain_openai import OpenAI, ChatOpenAI

class RagSystem:
    def __init__(self, data_config_path = 'config/model_config.yaml'):
        self.data_config_path = data_config_path
        self.data_config = config_parser(data_config_path = 'config/model_config.yaml')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.device = take_device()
        self.prompt = self.set_custom_prompt()
        self.llm = self.load_llm()
        self.embeddings = self.load_embeddings()
        self.qa_result= self.qa_bot()
        

    def load_embeddings(self):
        embeddings=HuggingFaceEmbeddings(model_name=self.data_config.get('embedding_model'),
                                        model_kwargs={'device':self.device})
        return embeddings

    def set_custom_prompt(self):
        '''
        Prompt template for QA retrieval for each vector store
        '''
        custom_prompt_template_path = self.data_config.get('custom_prompt_template_path')

        with open(custom_prompt_template_path, 'r') as file:
            custom_prompt_template = file.read()
            
        prompt = PromptTemplate(template=custom_prompt_template, 
                                input_variables=['context','question'])

        return prompt

    def load_chat_llm(self):
        llm = ChatOpenAI(model=self.data_config.get('openai_model'),
                        openai_api_key=self.OPENAI_API_KEY, 
                        max_tokens = self.data_config.get('openai_max_tokens')//2, #reduce the cost
                        temperature = self.data_config.get('openai_temperature'),
                        )
        return llm
        

    def load_llm(self):
        if self.data_config.get('openai_model')=="gpt-3.5-turbo-instruct":
            llm = OpenAI(model=self.data_config.get('openai_model'),
                        openai_api_key=self.OPENAI_API_KEY, 
                        max_tokens = self.data_config.get('openai_max_tokens'),
                        temperature = self.data_config.get('openai_temperature'),
                        )
        elif self.data_config.get('openai_model') in ["gpt-4o"]: # Will add more later
            llm=self.load_chat_llm()

        return llm

    def retrieval_qa_chain(self, vector_db):
        qa_chain=RetrievalQA.from_chain_type(
        llm= self.llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={'k':self.data_config.get('k_similar')}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':self.prompt }
        )
        return qa_chain

    def qa_bot(self):

        vector_db = FAISS.load_local(self.data_config.get('db_faiss_path'), 
                                    self.embeddings, 
                                    allow_dangerous_deserialization=True)
        qa_chain = self.retrieval_qa_chain(vector_db)                            
        return qa_chain 


    def final_result(self, query:str):
        response=self.qa_result({'query':query})
        return response 

    def update_vector_db(self):
        vector_db = VectorDatabase()
        documents, file_path_list = vector_db.create_vector_db() 
        print('file_path_list', file_path_list)
        self.qa_result= self.qa_bot()
        return 

def format_result(result):
    print(f"result: {result['result']}")
    for i, doc in enumerate(result['source_documents']):
        print(f'--page_content {i}: {doc.page_content}')
        print(f'--metadata {i}: {doc.metadata}')

if __name__ == "__main__":
    rag_system = RagSystem(data_config_path='config/model_config.yaml')
    rag_system.update_vector_db()
    while True:
        query=input("Enter your query: ")
        result = rag_system.final_result(query)
        print('result\n', result)
        print(format_result(result=result))


    # app.run(host='0.0.0.0', port=8083)

