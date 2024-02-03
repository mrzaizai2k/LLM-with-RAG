import sys
sys.path.append("")

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
# import chainlit as cl
import torch
from dotenv import load_dotenv
load_dotenv()

from src.utils import *
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from langchain_openai import OpenAI


from flask import Flask, request, jsonify


def set_custom_prompt():
    '''
    Prompt template for QA retrieval for each vector store
    '''
    prompt =PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

    return prompt

def load_llm():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, 
                        max_tokens = data.get('openai_max_tokens'),
                        temperature = data.get('openai_temperature'),
                        )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':data.get('k_similar')}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':prompt }
    )
    return qa_chain

def qa_bot():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':device})
    db = FAISS.load_local(db_faiss_path,embeddings)
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,db)
    return qa 


def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})
    return response 




app = Flask(__name__)

@app.route('/query', methods=['POST'])
def process_query():
    
    data = request.get_json()
    query = data.get('query', '')
    result = final_result(query)['result']
    return jsonify({'result': result})

if __name__ == "__main__":
    data = config_parser(data_config_path = 'config/model_config.yaml')

    db_faiss_path = data.get('db_faiss_path')
    custom_prompt_template_path = data.get('custom_prompt_template_path')

    with open(custom_prompt_template_path, 'r') as file:
        custom_prompt_template = file.read()

    device =  take_device()
    # while True:
    #     query=input("Enter your query: ")
    #     print(final_result(query))

    app.run(host='0.0.0.0', port=8083)




# ## chainlit here
# @cl.on_chat_start
# async def start():
#     chain=qa_bot()
#     msg=cl.Message(content="Firing up the company info bot...")
#     await msg.send()
#     msg.content= "Hi, welcome to company info bot. What is your query?"
#     await msg.update()
#     cl.user_session.set("chain",chain)


# @cl.on_message
# async def main(message):
#     chain=cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#     stream_final_answer=True, answer_prefix_tokens=["FINAL","ANSWER"]
#     )
#     cb.ansert_reached=True
#     # res=await chain.acall(message, callbacks=[cb])
#     res=await chain.acall(message.content, callbacks=[cb])
#     answer=res["result"]
#     sources=res["source_documents"]

#     if sources:
#         answer+=f"\nSources: "+str(str(sources))
#     else:
#         answer+=f"\nNo Sources found"

#     await cl.Message(content=answer).send() 