import sys
sys.path.append("")

import flask
from dotenv import load_dotenv
load_dotenv()

from src.utils import *
from src.deploy import *

from flask import Flask, request, jsonify


# app = Flask(__name__)

# @app.route('/query', methods=['POST'])
# def process_query():
    
#     data = request.get_json()
#     query = data.get('query', '')
#     result = final_result(query)
#     return jsonify({'result': result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8083)

import requests

# URL for the API endpoint
api_url = 'http://localhost:8083/query'

# Input query
query = input("Enter your query: ")

# Send POST request to the API
response = requests.post(api_url, json={'query': query})

# Print the response
print(response.json())
