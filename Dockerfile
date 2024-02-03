FROM ubuntu:latest


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install necessary packages
RUN make install && \
    apt-get update && \
    apt-get install -y make libreoffice python3 && \
    rm -rf /var/lib/apt/lists/*

# Make port 8083 available to the world outside this container
EXPOSE 8083

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run script.py when the container launches
CMD ["python", "src/deploy.py"]
