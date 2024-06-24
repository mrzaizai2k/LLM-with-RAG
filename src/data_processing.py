import sys
sys.path.append("")

import pymupdf
import os
import torch
from natsort import natsorted
import shutil
from transformers import pipeline
from src.utils import *

class MathLatexRecovery:
    def __init__(self, 
                 ocr_model=None, 
                 math_classifier=None, 
                 tmp_path="data/tmp/",
                 config_path:str= "config/model_config.yaml",):
         
        self.device=take_device()
        self.config_path = config_path
        self.config = config_parser(self.config_path)
        if ocr_model is None:
            ocr_model = pipeline("image-to-text", model=self.config["ocr_model"], 
                                 max_new_tokens=self.config["ocr_model_max_new_tokens"], device=self.device)
        if math_classifier is None:
            math_classifier = pipeline("text-classification", model=self.config["math_classifier"], 
                                       torch_dtype=torch.float16, device=self.device)
        
        self.ocr_model = ocr_model
        self.math_classifier = math_classifier
        self.tmp_path = tmp_path
        self.math_indices = []
        
    def is_latex_format(self, doc):
        # Check if doc.metadata exists and is a dictionary
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            return False
        
        # Check for "tex" in any metadata value (case-insensitive)
        return any("tex" in str(value).lower() for value in doc.metadata.values())

    def document_to_texts(self, documents)-> list:
        """Return the page content from documents of PyMuPDF"""
        return [txt.page_content for txt in documents]
    
    def get_math_indices(self, text_list, batch_size:int=8, 
                         truncation:str="only_first")-> list:
        outputs = self.math_classifier(text_list, batch_size=batch_size, truncation=truncation,)
        math_indices = [index for index, output in enumerate(outputs) if output['label'].lower() == "math"]
        return math_indices
    
        
    def save_math_doc_image(self, math_indices, pdf_path):
        doc = pymupdf.open(pdf_path)  # open document
        os.makedirs(self.tmp_path, exist_ok=True)
        image_paths = []
        for page in doc:  # iterate through the pages
            if not (page.number in math_indices):
                continue
            pix = page.get_pixmap()  # render page to an image
            save_path = f"{self.tmp_path}/%i.png" % page.number
            pix.save(save_path)  # store image as a PNG
            image_paths.append(save_path)
        return image_paths
    
    def remove_short_generated_texts(self, documents, threshold:int = 5)->list:
        """Remove elements with generated_text length less than the threshold"""
        filtered_documents = []

        for doc_list in documents:
            # Filter the list of dictionaries
            filtered_doc_list = [
                doc for doc in doc_list if len(doc['generated_text'].strip()) >= threshold
            ]
            # Add the filtered list to the final list if not empty
            if filtered_doc_list:
                filtered_documents.append(filtered_doc_list)

        return filtered_documents

    def get_math_ocr(self, image_list, batch_size=8,) -> list:
        ocr_texts = self.ocr_model(image_list, batch_size=batch_size)
        ocr_texts = self.add_index_to_gen_docs(ocr_texts)
        ocr_texts = self.remove_short_generated_texts(ocr_texts)
        return ocr_texts

    def update_documents(self, ori_documents, ocr_outputs:list):
        for i, item in enumerate(ocr_outputs):
            idx = int(item[0]['index'])
            ori_documents[idx].page_content = ocr_outputs[i][0]["generated_text"]
        return ori_documents

    def add_index_to_gen_docs(self, ocr_texts):
        for i, index in enumerate(self.math_indices):
            for item in ocr_texts[i]:
                item['index'] = index
        return ocr_texts
    
    def recover_math(self, documents):
        self.math_indices = []
        pdf_path = documents[0].metadata["source"]
        text_list = self.document_to_texts(documents=documents)
        self.math_indices = self.get_math_indices(text_list=text_list,batch_size=8, truncation="only_first")
        if not self.math_indices:
            return documents
        print(f"There are {len(self.math_indices)} / {len(text_list)} pages has math")
        image_paths = self.save_math_doc_image(math_indices=self.math_indices, pdf_path=pdf_path)
        ocr_texts = self.get_math_ocr(image_list = image_paths, batch_size=8)
        outputs = self.update_documents(ori_documents=documents, ocr_outputs=ocr_texts)
        remove_and_recreate_folder(folder_path=self.tmp_path)
        return outputs
        

if __name__ == "__main__":
    pdf_path = "data/web_data/Lecture 3.pdf"
    loader = PyMuPDFLoader(pdf_path)
    ori_text = loader.load()
    ori_text, duplicate_count = remove_duplicate_documents(ori_text)
    post_processor = MathLatexRecovery()
    outputs = post_processor.recover_math(ori_text[:15])
    combined_docs = combine_short_doc(outputs, 100)
    print("outputs[:5]",outputs[:5])