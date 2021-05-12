FROM ubuntu:18.04

RUN apt-get update -y && apt-get install -y python3-pip python3-dev git gcc g++ ffmpeg libsm6 libxext6

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 80

CMD ["python3", "app.py"]
