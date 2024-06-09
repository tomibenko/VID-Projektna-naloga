# Use the official TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Install build-essential and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy the application files into the container
COPY . /app


# Install Python dependencies
RUN pip3 install --ignore-installed -r requirements.txt

# Expose the application port
EXPOSE 8080

# Set the command to run the application
CMD ["python3", "data_managing.py"]
