FROM --platform=linux/arm64 python:3.10-slim-buster

# install git and ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg

WORKDIR /app
RUN git clone https://github.com/smpurkis/cat-cam.git .
WORKDIR /app/cat-cam
RUN pip install -r requirements.txt
CMD ["python", "discord_server.py"]