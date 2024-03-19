

import torch
import torch.nn as nn

import xformers.ops as xops
from xformers.ops import fmha

from model_architecture.cnn_embedder import CNNEmbedder


class XMHA(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_bias=None):
        x = self.layernorm1(x)
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = q.reshape(*q.shape[:2], self.n_heads, -1).transpose(1,2)
        k = k.reshape(*k.shape[:2], self.n_heads, -1).transpose(1,2)
        v = v.reshape(*v.shape[:2], self.n_heads, -1).transpose(1,2)
        a = xops.fmha.memory_efficient_attention(q, k, v, p=0.1, attn_bias=attn_bias)
        a = a.transpose(1,2)
        a = a.reshape(*a.shape[:2], -1)
        x_norm = self.layernorm2(x + a)
        x = x + self.ff(x_norm)
        return x

class XModel(nn.Module):
    def __init__(self, n_bbpe, n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.0):
        super().__init__()
        self.n_bbpe = n_bbpe
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.window = 48

        # Encoder embeddings
        self.cnnemb = CNNEmbedder(d_model)#, N=64, n=128, ff=d_model, first_k=3, first_s=2, last_s=1)
        # Encoder
        self.encoder = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.enc2bbpe = nn.Linear(d_model, n_bbpe)
    
    def forward(self, mels_tensor, mel_lens, bbpes_tensor):
        # enc emb
        mels_tensor = mels_tensor.unsqueeze(1)
        src, src_lens = self.cnnemb(mels_tensor, mel_lens)
        # enc
        attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=self.window-1, window_right=0)
        for enc in self.encoder:
            src = enc(src, attn_bias)
        enc_pred = self.enc2bbpe(src)
        return enc_pred, src_lens