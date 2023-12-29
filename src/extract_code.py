import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from pathlib import Path
import pickle
import os

def get_all_links_this_page(url,main_web_url,dicnry_links):
    if str(url).endswith(".pdf"):
        return
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all("a", href=True):
        if str(link["href"]).startswith("/"):
            link_full=main_web_url+link["href"]
#             print(link_full)    
            if link_full not in dicnry_links:
                dicnry_links[link_full]=0

dicnry_links={}
main_web_url='https://www.udst.edu.qa'
url = 'https://www.udst.edu.qa'
dicnry_links[url]=0
counter=0

while True:
    try:
        # get all un-visited web pages
        un_visited_urls=[]
        for url,item in dicnry_links.items():
            if item==0:
                un_visited_urls.append(url)
        for url in un_visited_urls:
            get_all_links_this_page(url,main_web_url,dicnry_links)
            dicnry_links[url]=1
            counter+=1
            print("Completed for link:",url)
            print("Completed",counter,"out of",len(list(dicnry_links.values())))
        if all(list(dicnry_links.values())):
            break
    except requests.exceptions.SSLError:
        print(url,"sslerror")
        dicnry_links[url]=1
        pass
    except requests.exceptions.HTTPSConnectionPool:
        print(url,"HTTPSConnectionPool")
        dicnry_links[url]=1
        pass

destination_dir="web_data"
dest_f=main_web_url.split("://")[-1]
print(dest_f)
destination_dir=destination_dir+'/'+dest_f.replace(".","_")
print(destination_dir)
if not os.path.isdir(destination_dir):
    os.mkdir(destination_dir)

# now go through the link and download pages as html or pdf

for url in dicnry_links.keys():
    try:
        print(url)
        if str(url).endswith(".pdf") or str(url).endswith(".PDF"):
            url=str(url).lower()
            file_name=url.split(".pdf")[-2]
            file_name=file_name.split("/")[-1]
            file_name=file_name+".pdf"
            filename = Path(destination_dir+"/"+file_name)
            response = requests.get(url)
            filename.write_bytes(response.content)
    
            print("pdf file name",file_name)
            continue
        response = requests.get(url)  
        file_name=url+".html"
        file_name=file_name.replace("/","_")
        print(file_name)
        with open(destination_dir+"/"+file_name,"w") as f:
            f.write(response.text)
    except:
        print("issue")
        pass