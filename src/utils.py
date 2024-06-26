import sys
sys.path.append("")
import os
import  torch
import yaml
import time 
import shutil
import re
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
    print(f"model_type: {result['model_type']}")
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

def remove_duplicate_documents(documents):
    """Deduplicate documents by their page content"""
    unique_contents = set()
    unique_documents = []
    duplicate_count = 0

    for doc in documents:
        content = doc.page_content
        if content not in unique_contents:
            unique_contents.add(content)
            unique_documents.append(doc)
        else:
            duplicate_count += 1
    if duplicate_count > 0:
        print(f"Number of duplicate documents removed: {duplicate_count}")
    return unique_documents, duplicate_count

def remove_short_documents(documents, threshold:int=3):
    """Remove documents with page content length less than the threshold"""
    filtered_documents = []
    removed_count = 0

    for doc in documents:
        if len(doc.page_content) >= threshold:
            filtered_documents.append(doc)
        else:
            removed_count += 1

    print(f"Number of short documents removed: {removed_count}")
    return filtered_documents

def remove_and_recreate_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
    # Recreate the folder
    os.makedirs(folder_path)

def is_latex_format(doc):
    # Check if doc.metadata exists and is a dictionary
    if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
        return False
    
    # Check for "tex" in any metadata value (case-insensitive)
    return any("tex" in str(value).lower() for value in doc.metadata.values())

def remove_repetitive_patterns(text:str):
    # Define regex patterns to match repetitive sequences
    patterns = [
        r'(.\n\n)+',                      # Matches repetitive sequences of ".\n"
        r'(\\\(T\\.\\\)\\\(\\.\\\))+',   # Matches repetitive sequences of "\(T.\)\(.\)"
        r'(\w+)\1+',                      # Matches repetitive sequences of a word
        r'(\\!)+',
    ]
    
    # Process each pattern and replace it with a single instance
    for pattern in patterns:
        text = re.sub(pattern, r'\1', text)
    
    return text

def find_common_prefix(strings, threshold=0.7):
    if not strings:
        return ""

    # Count occurrences of each prefix
    prefix_counts = {}
    total_strings = len(strings)

    for s in strings:
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            if prefix in prefix_counts:
                prefix_counts[prefix] += 1
            else:
                prefix_counts[prefix] = 1

    # Calculate the minimum occurrence threshold
    min_occurrences = int(total_strings * threshold)

    # Find the longest prefix that appears at least `min_occurrences` times
    common_prefix = ""
    for prefix, count in prefix_counts.items():
        if count >= min_occurrences and len(prefix) > len(common_prefix):
            common_prefix = prefix

    return common_prefix


def remove_common_prefix_from_documents(documents, 
                                        max_len_documents:int = 200,
                                        common_prefix_threshold:float = 0.7):
    """
    Removes the longest common prefix from a list of documents, optimizing for efficiency with large datasets.

    This function aims to streamline document processing by removing any repeated introductory text (header) shared across a majority (default 70%) of the documents. This can improve subsequent analysis or summarization tasks by focusing on the unique content within each document.

    **Parameters:**

    - documents (list): A list of document objects, each likely containing a `page_content` attribute holding textual content.
    - max_len_documents (int, optional): A threshold to limit processing for large datasets. If the number of documents exceeds this threshold, the original documents are returned unmodified (default is 200). This helps prevent performance issues when working with extensive collections.

    **Returns:**

    - list: The modified list of documents with their common prefix removed, if a prefix was found and removed. Otherwise, the original documents are returned.

    **Note:**

    - The specific mechanism for accessing document content (`page_content` in this example) may vary depending on your document object implementation. Please adjust the attribute name accordingly.
    """
    if len(documents) > max_len_documents:
        return documents
    
    # Extract page content from each document
    strs = [doc.page_content for doc in documents]

    # Find the longest common prefix
    common_prefix = find_common_prefix(strs, threshold=common_prefix_threshold)
    # Remove the common prefix from each document's content
    for doc in documents:
        doc.page_content = doc.page_content[len(common_prefix):]

    return documents
