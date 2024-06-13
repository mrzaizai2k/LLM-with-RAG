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


@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query', '')
    result = rag_system.final_result(query)
    result['source_documents']=[serialize_document(doc) for doc in result['source_documents']] # list(dict)

    return jsonify(result), 200



@app.route('/update', methods=['POST'])
def update_db():
    try: 
        # Assuming rag_system is defined somewhere else in your code 
        rag_system.update_vector_db()
        # Returning a response with JSON format for better clarity
        return jsonify({'message': 'Update successful'}), 200
    except Exception as e:
        # Returning a meaningful error code and message in case of exceptions
        return jsonify({'message': 'Internal server error: {}'.format(str(e))}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8083)