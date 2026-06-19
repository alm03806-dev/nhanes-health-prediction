"""
FT-Transformer architecture.

This is copied verbatim from the training notebook (nhanesv5.ipynb).
It MUST stay identical to the training-time definition, or the saved
state_dict (ft_state_dict.pt) will fail to load / will load incorrectly.
"""
import math
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_token, n_heads, attn_dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_token, 3 * d_token, bias=False)
        self.proj = nn.Linear(d_token, d_token)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_factor=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = MultiHeadSelfAttention(d_token, n_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor), nn.GELU(),
            nn.Dropout(ffn_dropout), nn.Linear(d_token * ffn_factor, d_token))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=192, n_heads=8, n_layers=3,
                 ffn_factor=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_factor, attn_dropout, ffn_dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0])).squeeze(-1)


def get_ft_probs(model, X_np, batch_size=512, device="cpu"):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            b = torch.tensor(X_np[i:i + batch_size], dtype=torch.float32).to(device)
            probs.extend(torch.sigmoid(model(b)).cpu().numpy())
    import numpy as np
    return np.array(probs)


def get_svm_probs(model, X):
    """Min-max normalized decision_function — matches training-time scoring."""
    import numpy as np
    scores = model.decision_function(X)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores
