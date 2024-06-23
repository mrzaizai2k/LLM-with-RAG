import sys
sys.path.append("")
import os
import  torch
import yaml
import time 
from natsort import natsorted
from langchain_community.document_loaders import PyMuPDFLoader
                                                 


def serialize_document(document):
    return {
        'page_content': document.page_content,
        'metadata': document.metadata
    }

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper

def is_file(path: str):
    return '.' in path

def check_path(path):
    # Extract the last element from the path
    last_element = os.path.basename(path)
    if is_file(last_element):
        # If it's a file, get the directory part of the path
        folder_path = os.path.dirname(path)

        # Check if the directory exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create new folder path: {folder_path}")
    else:
        # If it's not a file, it's a directory path
        # Check if the directory exists, create it if not
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create new path: {path}")

def take_device():
    # Check for GPU availability
    gpu_available = torch.cuda.is_available()

    # Set the device based on availability
    device = torch.device("cuda" if gpu_available else "cpu")

    # Print the selected device
    print(f"Selected device: {device}")

    return device

def config_parser(data_config_path = 'config/config.yaml'):
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def format_result(result):
    print(f"result: {result['result']}")
    for i, doc in enumerate(result['source_documents']):
        print(f'--page_content {i}: {doc.page_content}')
        print(f'--metadata {i}: {doc.metadata}')

def serialize_document(document):
    return {
        'page_content': document.page_content,
        'metadata': document.metadata
    }

def combine_short_doc(ori_text, threshold:int = 100):
    """Combine short doc into long doc and remove short doc for larger context"""
    i = 0
    while i < len(ori_text) - 1:
        if len(ori_text[i].page_content) < threshold:
            ori_text[i + 1].page_content = ori_text[i].page_content + " " + ori_text[i + 1].page_content
            del ori_text[i]
        else:
            i += 1

    # Ensure the last item also meets the threshold requirement
    if len(ori_text[-1].page_content) < threshold and len(ori_text) > 1:
        ori_text[-2].page_content += " " + ori_text[-1].page_content
        del ori_text[-1]

    return ori_text

def get_all_dir(root_dir, sort=True):
    # Get the list of items in the directory
    items = os.listdir(root_dir)
    # Sort the items list in natural order
    if sort:
        items = natsorted(items)
    # Create full paths
    image_list = [os.path.join(root_dir, item) for item in items]
    return image_list
