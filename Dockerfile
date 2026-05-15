FROM python:3.11-slim

# ffmpeg is required for clip extraction (video_utils.extract_clip).
# Neither Railpack nor Nixpacks reliably installs it across Railway plans,
# so we pin it explicitly here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache across rebuilds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of the repo.
COPY . .

# Railway injects $PORT at runtime — Streamlit binds to it.
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false

CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0
