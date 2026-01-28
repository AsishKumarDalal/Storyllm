from ..src.model.llm import GPT
from pathlib import Path
import yaml
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DICT=ROOT_DIR/ "configs" / "model.yaml"
with open(MODEL_CONFIG_DICT) as f:
    GPT_CONFIG_124M= yaml.safe_load(f)

model=GPT(GPT_CONFIG_124M)
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    return total

total_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
