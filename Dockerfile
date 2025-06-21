FROM python:3.10-slim
WORKDIR /app
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt
copy app.py .
copy model.pkl .
EXPOSE 5050
CMD ["python","app.py"]
