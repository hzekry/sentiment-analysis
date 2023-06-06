FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the Flask app files to the working directory
COPY . /app

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

# Download the necessary NLTK data
RUN python -m nltk.downloader stopwords

# Set the entrypoint command to run the Flask app
CMD ["python", "SA.py"]
