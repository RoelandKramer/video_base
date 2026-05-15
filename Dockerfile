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

# Railway injects $PORT at runtime — Streamlit must bind to it AND 0.0.0.0.
# Use `sh -c` so the shell expands $PORT; exec-form CMD (JSON array) doesn't
# perform variable expansion.
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Unset STREAMLIT_SERVER_PORT before launching — some Railway projects inject
# this as the literal string "$PORT" via shared/project variables, which
# Streamlit then can't parse. After unsetting, the --server.port arg wins.
CMD ["sh", "-c", "unset STREAMLIT_SERVER_PORT && streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
