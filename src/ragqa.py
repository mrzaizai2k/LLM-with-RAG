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
from sentence_transformers import SentenceTransformer, util
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import TavilySearchAPIRetriever, BM25Retriever

from Utils.utils import *

class RagSystem:
    def __init__(self, data_config_path = 'config/model_config.yaml'):
        self.data_config_path = data_config_path
        self.data_config = config_parser(data_config_path = 'config/model_config.yaml')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.TAVILY_KEY = os.getenv('TAVILY_KEY')
        self.device = take_device()
        self.prompt = self.set_custom_prompt()
        self.routing_model = self.load_model_routing()
        self.rerank_model = self.load_rerank_model()
        self.embeddings = self.load_embeddings()
        self.vector_db = self.load_vector_db()

        self.embedding_retriever = self.load_embedding_retriever()
        self.bm25_retriever = self.load_bm25_retriever()
        self.ensemble_retriever = self.load_ensemble_retriever()
        self.web_retriever = self.load_web_retriever()

        
    def load_vector_db(self):
        return FAISS.load_local(self.data_config.get('db_faiss_path'), 
                                    self.embeddings, 
                                    allow_dangerous_deserialization=True)

    def load_model_routing(self):
        routing_model=SetFitModel.from_pretrained(self.data_config["routing_model"], force_download=False).to(self.device)
        routing_model.predict(["who is karger"]) #load the dummy input
        return routing_model
    
    def load_rerank_model(self):
        rerank_model = SentenceTransformer(self.data_config.get('rerank_model'), device= self.device)
        return rerank_model


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
                        max_tokens = self.data_config.get('openai_max_tokens'), #reduce the cost
                        temperature = self.data_config.get('openai_temperature'),
                        )

        return llm, model_type

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def load_embedding_retriever(self):
        retriever=self.vector_db.as_retriever(search_kwargs={'k':self.data_config['k_similar'], 
                                                        "search_type": self.data_config['search_type'],
                                                        "lambda_mult": self.data_config['lambda_mult'],
                                                        })
        return retriever

    def load_bm25_retriever(self):
        documents = get_documents_from_vectordb(self.vector_db)
        bm25_retriever = BM25Retriever.from_documents(documents = documents,)
        bm25_retriever.k=self.data_config['k_similar']
        return bm25_retriever  
    
    def load_ensemble_retriever(self):
        bm25_weight = self.data_config['bm25_weight']
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.embedding_retriever], 
            weights=[bm25_weight, 1-bm25_weight]
        )
        return ensemble_retriever
        
    def load_web_retriever(self):
        retriever = TavilySearchAPIRetriever(k=2, api_key=self.TAVILY_KEY)
        return retriever

    def retrieval_qa_chain(self, llm, prompt):

        chain = (
             prompt
            | llm
            | StrOutputParser()
        )
        return chain

    @timeit
    def rerank_document_similarity(self, query, ori_documents, n_docs:int = None):
        documents = ori_documents.copy()
        docs_txt_list  = [txt.page_content for txt in documents]
        query_embedding = self.rerank_model.encode(query)
        passage_embedding = self.rerank_model.encode(docs_txt_list)

        similarity_scores = util.dot_score(query_embedding, passage_embedding)
        similarity_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
        print(f"sorted_indices: {sorted_indices}")

        # Sort the original list of texts based on the sorted indices
        rerank_documents = [documents[idx] for idx in sorted_indices.tolist()[0]]
        if n_docs:
            rerank_documents = rerank_documents[:n_docs]
        return similarity_scores, rerank_documents

    def _format_result(self, query, source_documents, answer, model_type):
        response={}
        response['query'] = query
        response['source_documents'] = source_documents
        response['result'] = answer
        response['model_type'] = model_type
        return response

    def final_result(self, query:str):
        llm, model_type = self.load_llm(query)

        if model_type != "gpt-3.5-turbo-instruct":
            print("use rag fusion")

        relevant_docs = self.ensemble_retriever.get_relevant_documents(query)

        similarity_scores, rerank_documents = self.rerank_document_similarity(query = query, ori_documents=relevant_docs, 
                                                           n_docs=self.data_config['k_similar_rerank'])
        
        if similarity_scores[0][0] < self.data_config['web_search_thresh']: #check max score
            web_docs = self.web_retriever.invoke(query)
            rerank_documents.extend(web_docs)

        qa_chain = self.retrieval_qa_chain(llm, self.prompt)  
        result= qa_chain.invoke({"question": query, "context": self.format_docs(rerank_documents)})
        response = self._format_result(query, rerank_documents, result, model_type) 
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

