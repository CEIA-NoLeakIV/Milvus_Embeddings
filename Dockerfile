FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ca-certificates \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.insightface/models \
    && curl -fsSL -o /root/.insightface/models/scrfd_10g_bnkps.onnx \
        https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx \
    && echo "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91  /root/.insightface/models/scrfd_10g_bnkps.onnx" \
        | sha256sum -c -

COPY . .

CMD ["python", "failure_analysis.py", "--help"]
