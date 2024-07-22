import sys
sys.path.append("")

from dotenv import load_dotenv
load_dotenv()

import warnings; 
warnings.filterwarnings("ignore")

from Utils.utils import *
from src.ragqa import RagSystem
from flask import Flask, request, jsonify
from src.Utils.logger import create_logger
logger = create_logger()

app = Flask(__name__)

rag_system = RagSystem(data_config_path='config/model_config.yaml')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query', '')
    result = rag_system.final_result(query)
    print(format_result(result=result))
    logger.debug(msg = f"{result}")
    result['source_documents']=[serialize_document(doc) for doc in result['source_documents']] # list(dict)
    return jsonify(result), 200


@app.route('/update', methods=['POST'])
def update_db():
    try: 
        # Assuming rag_system is defined somewhere else in your code 
        rag_system.update_vector_db()
        # Returning a response with JSON format for better clarity
        msg = 'Update successful'
        logger.debug(msg)
        return jsonify({'message': msg}), 200
    except Exception as e:
        # Returning a meaningful error code and message in case of exceptions
        msg = f'Internal server error: {e}'
        logger.debug(msg)
        return jsonify({'message': f'Internal server error: {e}'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8083)