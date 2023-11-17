FROM python:3.8-alpine
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "-m", "streamlit", "run", "ui/app.py"]