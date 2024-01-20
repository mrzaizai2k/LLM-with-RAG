import sys
sys.path.append("")
import os
import  torch
import yaml
import time 

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