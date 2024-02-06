import sys
sys.path.append("")

from dotenv import load_dotenv
load_dotenv()

from src.utils import *
from src.ragqa import *


from flask import Flask, request, jsonify


app = Flask(__name__)

rag_system = RagSystem(data_config_path='config/model_config.yaml')

def serialize_document(document):
    return {
        'page_content': document.page_content,
        'metadata': document.metadata
    }

def format_result(result):
    for i, doc in enumerate(result['source_documents']):
        print(f'--page_content {i}: {doc.page_content}')
        print(f'--metadata {i}: {doc.metadata}')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query', '')
    result = rag_system.final_result(query)
    result['source_documents']=[serialize_document(doc) for doc in result['source_documents']] # list(dict)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8083)