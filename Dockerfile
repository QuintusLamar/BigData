FROM python:3.8
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY . .
EXPOSE 8080
CMD ["python3", "-m", "streamlit", "run", "ui/app.py"]