import torch
from pathlib import Path
from ..src.model.llm import GPT
import yaml
from ..src.tokenizer.bpe import enc
tokenizer=enc
ROOT_DIR = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from ..src.inference.generate import *
MODEL_CONFIG_DICT=ROOT_DIR/ "configs" / "model.yaml"
with open(MODEL_CONFIG_DICT) as f:
    GPT_CONFIG_124M= yaml.safe_load(f)
checkpoint = torch.load(f"{ROOT_DIR}/checkpoint.pth", map_location=device)  # or "cuda"
model=GPT(GPT_CONFIG_124M).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

generate_and_print_sample__(model,tokenizer,device,"sam was studing ")