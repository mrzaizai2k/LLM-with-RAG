import sys
sys.path.append("")

from dotenv import load_dotenv
load_dotenv()

from src.utils import *
from src.ragqa import *

import requests

def test_llm():
    # URL for the API endpoint
    api_url = 'http://localhost:8083/query'

    # Input query
    query = input("Enter your query: ")

    # Send POST request to the API
    response = requests.post(api_url, json={'query': query})

    # Print the response
    print(response.json())
    return response.json()

def test_update_db():
    url = 'http://localhost:8083/update'  # Update the URL if your Flask app runs on a different port or host

    try:
        response = requests.post(url)
        if response.status_code == 200:
            print("API Test Successful: Update was successful")
        else:
            print(f"API Test Failed: {response.status_code} - {response.json()['message']}")
    except Exception as e:
        print(f"API Test Failed: {str(e)}")

if __name__ == '__main__':
    test_update_db()
    test_llm()
