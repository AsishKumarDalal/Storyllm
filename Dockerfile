FROM python:3.10-slim

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# System dependencies (git needed for DVC)
# -------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
COPY custom-llm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc 

# -------------------------
# Copy ONLY necessary files
# -------------------------
COPY custom-llm/src/ src/
COPY custom-llm/configs/ configs/
COPY custom-llm/serve/ serve/

# DVC metadata (for pulling model only)
COPY .dvc/ .dvc/
COPY .dvcignore .
COPY custom-llm/checkpoint.pth.dvc checkpoint.pth.dvc

# -------------------------
# Initialize git (DVC requires it)
# -------------------------
RUN git init && \
    git config user.email "docker@build.local" && \
    git config user.name "Docker Build"

# -------------------------
# Configure DVC remote
# -------------------------
RUN dvc remote list | grep -q dagshub || dvc remote add -d dagshub https://dagshub.com/"${DAGSHUB_USER}"/"{REPO}".dvc
RUN dvc remote modify dagshub auth basic

# -------------------------
# Pull model checkpoint using env vars
# -------------------------
ARG DAGSHUB_USER
ARG DAGSHUB_PASSWORD

RUN dvc remote modify dagshub --local user "${DAGSHUB_USER}" && \
    dvc remote modify dagshub --local password "${DAGSHUB_PASSWORD}" && \
    dvc status checkpoint.pth.dvc && \
    dvc pull checkpoint.pth.dvc -v

# -------------------------
# Expose API port
# -------------------------
EXPOSE 8000

# -------------------------
# Start FastAPI server
# -------------------------
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]