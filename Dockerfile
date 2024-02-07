FROM python:3.10
# FROM mrzaizai2k/llm_n_rag


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install -r setup.txt 
# Install necessary packages
RUN apt-get update && \
    apt-get install -y make libreoffice python3 && \
    rm -rf /var/lib/apt/lists/*

# Make port 8083 available to the world outside this container
EXPOSE 8083

# ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run script.py when the container launches
CMD ["python", "src/api.py"]
