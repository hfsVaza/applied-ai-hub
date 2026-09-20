# AI Projects Hub — CPU-only container.
#
#   docker compose up --build        then open http://localhost:7860
#
# Models are not baked into the image. They download on first use into the
# huggingface cache volume, so the first run of each project is slow and every
# run after it is fast, exactly as on a local install.

FROM python:3.11-slim

# OpenCV and MediaPipe need these shared libraries even when nothing is drawn
# on screen — MediaPipe's native library links against EGL and GLES whether or
# not it renders. ffmpeg gives OpenCV a dependable video decoder and encoder.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    HUB_HEADLESS=1 \
    HUB_HOST=0.0.0.0 \
    HUB_PORT=7860

WORKDIR /app

# Torch comes from the CPU-only index. The default PyPI wheels carry a full
# CUDA runtime, which adds several gigabytes that a CPU container never uses.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch torchvision \
    && pip install -r requirements.txt

COPY . .

# Everything the app writes at runtime, owned by the unprivileged user it runs
# as. These are the paths docker-compose.yml mounts volumes over.
RUN useradd --create-home --uid 1000 hub \
    && mkdir -p /app/output/anonymizer /app/output/depth /app/output/pose \
                /app/semantic_search/data/library \
                /cache/huggingface /cache/torch \
    && chown -R hub:hub /app /cache

USER hub

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/status', timeout=4)"

CMD ["python", "hub.py"]
