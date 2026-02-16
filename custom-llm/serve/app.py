import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from src.model.llm import GPT
from src.tokenizer.encode_decode import text_to_token_ids, token_ids_to_text
import yaml
from src.tokenizer.bpe import enc
tokenizer=enc

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DICT=ROOT_DIR/ "configs" / "train.yaml"
MODEL_CONFIG_DICT=ROOT_DIR/ "configs" / "model.yaml"
with open(MODEL_CONFIG_DICT) as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Load model
# ----------------------
checkpoint = torch.load(f"{ROOT_DIR}/checkpoint.pth", map_location=device) 
model = GPT(cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# tokenizer (example: tiktoken)


# ----------------------
# API
# ----------------------
app = FastAPI()

class Prompt(BaseModel):
    text: str
    max_new_tokens: int = 100

@app.post("/generate")
def generate_text(prompt: Prompt):
    input_ids = text_to_token_ids(prompt.text, tokenizer).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=prompt.max_new_tokens)
    output_text = token_ids_to_text(output_ids, tokenizer)
    return {"output": output_text}
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "device": str(device)
    }
    