FROM python:3.11-slim

WORKDIR /app

# torch bản CPU cài trước: để pip tự chọn thì sẽ kéo bản CUDA, image nặng thêm vài GB
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY OriginalDocs/ OriginalDocs/
COPY assets/ assets/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
