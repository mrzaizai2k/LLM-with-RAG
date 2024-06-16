import sys
sys.path.append("")

import time
import warnings
warnings.filterwarnings("ignore")
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

from src.ingest import VectorDatabase
from langchain_openai import OpenAI, ChatOpenAI
from setfit import SetFitModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils import *

class RagSystem:
    def __init__(self, data_config_path = 'config/model_config.yaml'):
        self.data_config_path = data_config_path
        self.data_config = config_parser(data_config_path = 'config/model_config.yaml')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.device = take_device()
        self.prompt = self.set_custom_prompt()
        self.routing_model = self.load_model_routing()
        self.embeddings = self.load_embeddings()
        self.vector_db = self.load_vector_db()
        
    def load_vector_db(self):
        return FAISS.load_local(self.data_config.get('db_faiss_path'), 
                                    self.embeddings, 
                                    allow_dangerous_deserialization=True)

    def load_model_routing(self):
        routing_model=SetFitModel.from_pretrained(self.data_config["routing_model"]).to(self.device)
        routing_model.predict(["who is karger"]) #load the dummy input
        return routing_model
    

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

    def load_llm(self, query:str):
        model_type = self.data_config.get('openai_model')[self.routing_model.predict([query])[0]]

        if model_type=="gpt-3.5-turbo-instruct":
            llm = OpenAI(model=model_type,
                        openai_api_key=self.OPENAI_API_KEY, 
                        max_tokens = self.data_config.get('openai_max_tokens'),
                        temperature = self.data_config.get('openai_temperature'),
                        )
        else:
            llm = ChatOpenAI(model=model_type,
                        openai_api_key=self.OPENAI_API_KEY, 
                        max_tokens = self.data_config.get('openai_max_tokens')//2, #reduce the cost
                        temperature = self.data_config.get('openai_temperature'),
                        )

        return llm, model_type

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def load_retriever(self):
        retriever=self.vector_db.as_retriever(search_kwargs={'k':self.data_config['k_similar'], 
                                                        "search_type": self.data_config['search_type'],
                                                        "lambda_mult": self.data_config['lambda_mult'],
                                                        })
        return retriever
    

    def retrieval_qa_chain(self, llm):

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
            | self.prompt
            | llm
            | StrOutputParser()
        )

        retrieve_docs = (lambda x: x["question"]) | self.load_retriever()

        chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
            answer=rag_chain_from_docs
        )
        return chain


    def final_result(self, query:str):
        response={}
        llm, model_type = self.load_llm(query)
        qa_chain = self.retrieval_qa_chain(llm)  
        result= qa_chain.invoke({"question":f"{query}"})
        response['query'] = query
        response['source_documents'] = result['context']
        response['result'] = result['answer']
        response['model_type'] = model_type
        return response 

    def update_vector_db(self):
        vector_db = VectorDatabase()
        _, file_path_list = vector_db.create_vector_db() 
        print('file_path_list', file_path_list)
        self.vector_db = self.load_vector_db()
        return 


if __name__ == "__main__":
    rag_system = RagSystem(data_config_path='config/model_config.yaml')
    # rag_system.update_vector_db()
    while True:
        query=input("Enter your query: ")
        result = rag_system.final_result(query)
        # print('result\n', result)
        print(format_result(result=result))


    # app.run(host='0.0.0.0', port=8083)

