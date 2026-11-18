FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY polarbench ./polarbench
COPY static ./static
ENV POLARBENCH_DATA_DIR=/data POLARBENCH_HOST=0.0.0.0 PORT=18943 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2
EXPOSE 18943
HEALTHCHECK --interval=8s --timeout=3s --start-period=5s --retries=3 CMD python3 -B -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18943/api/health')"
CMD ["python3", "-B", "app.py"]
