#StoryLLM: Custom GPT-Style Language Model

A custom-built, generative pre-trained transformer (GPT) implemented from scratch in PyTorch. 

👋 **Hello** 
Welcome to my repository! This project showcases my deep dive into Natural Language Processing (NLP) and Deep Learning.  built a decoder-only Large Language Model (LLM) completely from the ground up, moving from the foundational math of self-attention all the way to a production-ready API served with Docker and DVC.

---

## 🌟 Key Features

- **Built from Scratch Architecture**: A pure PyTorch implementation of the decoder-only Transformer architecture.
- **Core Components**: Custom implementation of Multi-Head Self-Attention, Multi-Layer Perceptrons (MLP), Layer Normalization, and Positional Embeddings.
- **Robust Training Pipeline**: Includes a unified training loop with learning rate scheduling, state checkpointing, and real-time visualization (loss vs. tokens curves).
- **Text Generation**: Autoregressive generation equipped with temperature sampling, Top-K filtering, and early stopping via EOT tokens.
- **Production-Ready Serving**: Integrated **FastAPI** backend to expose the model via standard REST API endpoints.
- **Modern MLOps**: CI/CD ready with **Docker** and Data Version Control (**DVC**) utilizing **DagsHub** for efficient model weight storage and retrieval.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch
- **Backend & Serving**: FastAPI, Uvicorn, Pydantic
- **MLOps & Deployment**: Docker, DVC (Data Version Control), DagsHub
- **Tokenization**: Byte-Pair Encoding (BPE) via OpenAI's `tiktoken`
- **Configuration**: YAML for model & training hyperparameters

---

## 📂 Project Structure

```text
Storyllm/
├── custom-llm/
│   ├── configs/       # YAML config files (model.yaml, train.yaml)
│   ├── src/           # Core source code
│   │   ├── model/     # Transformer blocks, attention, and GPT model classes
│   │   ├── tokenizer/ # BPE tokenization scripts
│   │   ├── training/  # Trainer logic, loss calculation, and schedulers
│   │   ├── inference/ # Text generation and sampling functionality
│   │   └── data/      # Dataset parsing and data collators
│   ├── scripts/       # Execution entry points (train.py, inference.py)
│   ├── serve/         # FastAPI endpoints (app.py)
│   └── requirements.txt
├── Dockerfile         # Docker configuration for inference deployment
├── .dvc/              # DVC configuration for managing model artifacts
└── template.py        # Automated project scaffolding script
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (CUDA supported and highly recommended for faster training/inference)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/Storyllm.git
cd Storyllm
```

2. Install the required Python dependencies:
```bash
pip install -r custom-llm/requirements.txt
```

### 🧠 Training the Model
Ready to train the LLM from scratch? The training script reads hyperparameters from `configs/train.yaml` and `configs/model.yaml` to instantiate the architecture. 

```bash
python -m custom-llm.scripts.train
```
*Note: This script will save the generated weights as `checkpoint.pth` and dynamically output training charts like `loss_curve.png`.*

### ✍️ Generating Text (Inference)
To test the model's creative capabilities within the terminal:

```bash
python -m custom-llm.scripts.inference
```

---

## 🕸️ API Serving

StoryLLM is ready to be integrated into any frontend or external application via a REST API.

Start the FastAPI development server:
```bash
uvicorn custom-llm.serve.app:app --host 0.0.0.0 --port 8000
```

### API Endpoints
- **Generate Text:** `POST /generate`  
  *Body:* `{"text": "Once upon a time", "max_new_tokens": 100}`
- **Health Check:** `GET /health`  
  *Returns server status and GPU context.*

---

## 🐳 Docker & MLOps Deployment

This project utilizes isolated Docker containers for deployment and relies on **DVC (Data Version Control)** connected to a **DagsHub** remote to pull the heavy model weights (`checkpoint.pth`) at runtime, keeping the Git repository lightweight.

```bash
# Build the Docker image
docker build -t storyllm .

# Run the container (Make sure to pass your DagsHub credentials)
docker run -p 8000:8000 \
  -e DAGSHUB_USER="your_username" \
  -e DAGSHUB_PASSWORD="your_password" \
  -e REPO="your_repo_name" \
  storyllm
```

---

## 🎯 To Recruiters and Engineering Managers

If you are reviewing this repository, this project demonstrates my ability to:
1. Understand and implement complex Deep Learning / Transformer mathematics from published papers.
2. Structure large Python codebases optimally (separating configurations, data pipelines, model architecture, and serving).
3. Think about the entire Machine Learning Lifecycle (MLOps), bridging the gap between training a model and serving it reliably in a containerized (Docker) environment handling artifact versions using DVC.



---

⭐️ **If you liked this project,  consider giving it a star on GitHub!**
