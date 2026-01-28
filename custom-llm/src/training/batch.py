import numpy as np
import torch
from pathlib import Path
import yaml
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DICT=ROOT_DIR/ "data" / "processed"
CONFIG_DICT=ROOT_DIR/ "configs" / "train.yaml"
with open(CONFIG_DICT) as f:
    cfg = yaml.safe_load(f)
if cfg["device"] == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = cfg["device"]

device_type = "cuda" if device == "cuda" else "cpu"
block_size=cfg["block_size"]
batch_size=cfg["batch_size"]

def get_batch(split):
    if split == 'train':
        data = np.memmap(f'{DATA_DICT}/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap(f'{DATA_DICT}/validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
