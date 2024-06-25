
import sys
sys.path.append("")

import os
from src.utils import config_parser
import pandas as pd

config = config_parser(data_config_path = 'config/gpt_routing_train_config.yaml')
dataset_name = config['dataset_name']
local_data_path = config['local_data_path']

file_name = os.path.basename(local_data_path).split(".")[0]
df = pd.read_excel(local_data_path)

print(df.head(10))
try:
    df.to_csv(f"hf://datasets/{dataset_name}/train.csv", index=False)
    print(f"Done pushing data to {dataset_name}")
except Exception as e:
    print(f"Error pushing data to {dataset_name}: {e}")