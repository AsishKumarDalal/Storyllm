from .attention import MultiHeadAttention
from .mlp import FeedForward
from .Layernorm import LayerNorm
import torch
import torch.nn as nn
import yaml
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
config_path = ROOT_DIR / "configs" / "model.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff= FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
    def forward(self,x):
        placeholder=x
        x= self.norm1(x)
        x=self.att(x)
        x=placeholder+x
        placeholder2=x
        x=self.norm2(x)
        x=self.ff(x)
        x=placeholder2+x
        return x


