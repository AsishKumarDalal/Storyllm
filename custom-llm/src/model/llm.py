import torch
import torch.nn as nn
from  .Layernorm import LayerNorm
from  .transformer import TransformerBlock
import yaml
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
config_path = ROOT_DIR / "configs" / "model.yaml"
EOT_TOKEN_ID=50256
with open(config_path) as f:
    cfg = yaml.safe_load(f)

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self,in_idx,y=None):
        batch_size, seq_len = in_idx.shape
        token_embedding=self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x=token_embedding+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm (x)
        x= self.out_head(x)
        return x

    @torch.no_grad
    def generate(self, idx, max_new_tokens=250, context_size=128, 
                         temperature=0.0, top_k=None, eos_id=EOT_TOKEN_ID):
                """
                Generate text with early stopping at EOT token.
                Stops immediately when EOT token is generated.
                """
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -context_size:]
                    
                    with torch.no_grad():
                        logits = self(idx_cond)
                    
                    logits = logits[:, -1, :]
                    
                    # Top-k filtering
                    if top_k is not None:
                        top_logits, _ = torch.topk(logits, top_k)
                        min_val = top_logits[:, -1]
                        logits = torch.where(
                            logits < min_val,
                            torch.tensor(float("-inf")).to(logits.device),
                            logits
                        )
                    
                    # Temperature sampling
                    if temperature > 0.0:
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    else:
                        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # STOP AT EOT TOKEN
                    if eos_id is not None and idx_next.item() == eos_id:
                        idx = torch.cat((idx, idx_next), dim=1)
                        break
                    
                    idx = torch.cat((idx, idx_next), dim=1)
                
                return idx
