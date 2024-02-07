# LLM-with-RAG
I just wanna build my own LLM with RAG

![CI/CD](https://github.com/mrzaizai2k/LLM-with-RAG/actions/workflows/workloads.yaml/badge.svg)
[![Repo Size](https://img.shields.io/github/repo-size/mrzaizai2k/LLM-with-RAG?style=flat-square)](https://github.com/mrzaizai2k/LLM-with-RAG)
[![License](https://img.shields.io/github/license/mrzaizai2k/LLM-with-RAG?style=flat-square)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/mrzaizai2k/LLM-with-RAG?style=flat-square)](https://github.com/mrzaizai2k/LLM-with-RAG/releases)

## Table of Contents
1. [Introduction](#introduction)
1. [Features](#features)
3. [How to Set Up](#how-to-set-up)
4. [Example Uses](#example-uses)
5. [To-Do List](#to-do-list)

## Introduction
Welcome to the my LLM with RAG system! This system is designed for me the ease the learning as a master in HCMUT

## Features
1. Update vector database

            curl -X POST http://localhost:8083/update

2. Ask questions with vector data

            curl -X POST -H "Content-Type: application/json" -d '{"query": "who is karger"}' http://localhost:8083/query

## How to Set Up

### Prerequisites
Before running the system, follow these steps to set up the environment:

1. **Clone the Repository:**
   - Close the Git repository to your local machine:
     ```bash
     git clone [repository_url]
     ```

2. **Install Dependencies:**
- Navigate to the project directory and install the required packages using the provided `setup.txt` file:
     ```bash
     pip install -r setup.txt
     ```
- To read ```.ppt``` file we need to run this code

        apt update
        apt install libreoffice 

3. **Get OPENAI_API_KEY Key:**
   - Google and get OPENAI_API_KEY from [OpenAI](https://openai.com/)

4. **Create .env File:**
   - Create a new file named `.env` in the project root directory.
   - Add the following line to the file
     ```env
     OPENAI_API_KEY=YOUR_OPENAI_API_KEY
     ```
## Run docker
For Linux you must open the port first:

      sudo ufw allow 8083

 docker:

      docker build -t mrzaizai2k/llm_n_rag .
      docker run -p 8083:8083 -v data:/app/data -e OPENAI_API_KEY llm_test
      

Build, run docker compose:

      docker-compose up

Test docker on port 8083:

      curl -X POST -H "Content-Type: application/json" -d '{"query": "who is karger"}' http://localhost:8083/query
      curl -X POST http://localhost:8083/update

## Example Uses

Explore practical implementations and demonstrations of the  functions in the `notebook` folder. These examples showcase real-world scenarios, illustrating how the chatbot can be effectively utilized for stock market monitoring.


## To-Do List

- [ ] **Update some features with langchain**
- [ ] **Build the docker to use with my Telegram bot**


