FROM ubuntu:latest


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN make bot
RUN sudo apt update
RUN sudo apt install libreoffice 

# Make port 8083 available to the world outside this container
EXPOSE 8083


# Run script.py when the container launches
CMD ["python", "src/deploy.py"]
