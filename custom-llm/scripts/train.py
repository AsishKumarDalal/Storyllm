import torch
import time
import yaml
from ..src.model.llm import GPT
from pathlib import Path
from ..artifacts.make import *
from ..src.training.trainer import *
from ..src.tokenizer.bpe import enc

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DICT=ROOT_DIR/ "configs" / "train.yaml"
MODEL_CONFIG_DICT=ROOT_DIR/ "configs" / "model.yaml"

with open(CONFIG_DICT) as f:
    cfg0= yaml.safe_load(f)
if cfg0["device"] == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = cfg0["device"]

with open(MODEL_CONFIG_DICT) as f:
    GPT_CONFIG_124M= yaml.safe_load(f)
device_type = "cuda" if device == "cuda" else "cpu"
block_size=cfg0["block_size"]
batch_size=cfg0["batch_size"]


tokenizer=enc

torch.manual_seed(123)
model = GPT(GPT_CONFIG_124M)
model.to(device)
start_time = time.time()
model.train()
num_epochs=cfg0["num_epochs"]
eval_freq=cfg0["eval_freq"]
eval_iter=cfg0["eval_iter"]
# Verify gradients are enabled (optional but good for debugging)
print("Checking if model parameters require grad...")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"WARNING: {name} does not require grad!")
        param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
print("running on",device);

train_losses, val_losses, tokens_seen = train_model_simple(
    model, optimizer, device,
    num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
    start_context="a boy got a ball", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    
}
torch.save(checkpoint, f'{ROOT_DIR}/checkpoint.pth')
save_training_charts(train_losses,val_losses,tokens_seen)