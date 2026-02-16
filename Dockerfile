FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY custom-llm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install dvc
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc

# -------------------------
# Copy project files
# -------------------------
COPY custom-llm/src/ src/
COPY custom-llm/configs/ configs/
COPY custom-llm/serve/ serve/

# DVC metadata only
COPY .dvc/ .dvc/
COPY .dvcignore .
COPY custom-llm/checkpoint.pth.dvc checkpoint.pth.dvc

# -------------------------
# Initialize git (required by DVC)
# -------------------------
RUN git init && \
    git config user.email "docker@runtime.local" && \
    git config user.name "Docker Runtime"

# -------------------------
# Expose port
# -------------------------
EXPOSE 8000

# -------------------------
# Start script (pull model at runtime)
# -------------------------
CMD sh -c "\
    dvc remote add -d dagshub https://dagshub.com/$DAGSHUB_USER/$REPO.dvc || true && \
    dvc remote modify dagshub auth basic && \
    dvc remote modify dagshub --local user $DAGSHUB_USER && \
    dvc remote modify dagshub --local password $DAGSHUB_PASSWORD && \
    dvc pull checkpoint.pth.dvc && \
    uvicorn serve.app:app --host 0.0.0.0 --port 8000 \
"
