FROM ubuntu:20.04
WORKDIR /app
RUN apt-get update && apt-get -y install build-essential
COPY . /app
EXPOSE 8080
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD ["python3", "projekt_1.py"]
