

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
        self.A2X = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, kv=None, attn_bias=None, kv_cache=None):
        residual = x
        x = self.layernorm1(x)
        if kv is None:
            kv = x
        q, k, v = self.Q(x), self.K(kv), self.V(kv)
        # Input tensors must be in format [B, M, H, K] where
        # B is the batch size
        # M the sequence length
        # H the number of heads and
        # K the embeding size per head
        q = q.reshape(*q.shape[:2], self.n_heads, -1)
        k = k.reshape(*k.shape[:2], self.n_heads, -1)
        v = v.reshape(*v.shape[:2], self.n_heads, -1)

        cached_k, cached_v = kv_cache['k'], kv_cache['v']
        if cached_k is not None:
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        kv_cache['k'], kv_cache['v'] = k, v

        a = xops.fmha.memory_efficient_attention(q, k, v, p=0.1*self.training, attn_bias=attn_bias)
        a = a.reshape(*a.shape[:2], -1)
        a = self.A2X(a)
        x = residual + a

        residual = x
        x = self.layernorm2(x)
        x = self.ff(x)
        x = residual + x

        return x, kv_cache

class XModel(nn.Module):
    def __init__(self, n_bbpe, n_layers=6, d_model=512, d_ff=2048, n_heads=8, window_size=48, dropout=0.0):
        super().__init__()
        self.n_bbpe = n_bbpe
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.window_size = window_size

        # Encoder embeddings
        self.cnnemb = CNNEmbedder(d_model)
        # BBPE embeddings
        self.bbpemb = nn.Embedding(n_bbpe, d_model)
        # Encoder
        self.encoder = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        # Cross-attention
        self.cross_attn = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        # Decoder
        self.decoder = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.enc_norm = nn.LayerNorm(d_model)
        self.enc2bbpe = nn.Linear(d_model, n_bbpe)
        self.dec_norm = nn.LayerNorm(d_model)
        self.dec2bbpe = nn.Linear(d_model, n_bbpe)
    
    def forward(self, mels_tensor, mel_lens, bbpes_tensor, kv_caches_enc=None, kv_caches_dec=None, n_cnn_processed=0, cross_mask=None):
        # enc emb
        mels_tensor = mels_tensor.unsqueeze(1)
        src, src_lens = self.cnnemb(mels_tensor, mel_lens, n_cnn_processed)
        # dec emb
        tgt = self.bbpemb(bbpes_tensor)
        # enc
        #attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=self.window_size-1, window_right=0)
        # https://facebookresearch.github.io/xformers/components/ops.html
        attn_bias = fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(_window_size=self.window_size)
        if kv_caches_enc is None: kv_caches_enc = [{'k':None, 'v':None} for _ in range(len(self.encoder))]
        if kv_caches_dec is None: kv_caches_dec = [{'k':None, 'v':None} for _ in range(len(self.decoder))]
        for i, layer in enumerate(self.encoder):
            # enc
            src, kv_cache = layer(src, attn_bias=attn_bias, kv_cache=kv_caches_enc[i])
            kv_caches_enc[i] = kv_cache
            # cross-attn  # TODO: dec and cross-attn swapped places ?
            # batchify
            # q_start = cross_mask.q_seqinfo.seqstart
            # q_len = q_start[1:] - q_start[:-1]
            # k_seqlen = cross_mask.k_seqinfo.seqlen
            # k_seqstart = cross_mask.k_seqinfo.seqstart
            # qs = torch.zeros(len(q_len), max(q_len), src.shape[-1], device=src.device, dtype=src.dtype)
            # for i in range(len(q_start)-1):
            #     qs[i, :q_len[i]] = src[:, q_start[i]:q_start[i+1]]
            # qs, _ = self.cross_attn[i](qs, kv=src, attn_bias=cross_mask, kv_cache={'k':None, 'v':None})
            # # unbatchify
            # qs = torch.zeros_like(tgt)

            q_starts = cross_mask.q_seqinfo.seqstart.tolist()
            k_starts = cross_mask.k_seqinfo.seqstart.tolist()
            if q_starts[-1] != tgt.shape[1]+1:
                print('q_lens[-1] != tgt.shape[1]+1')
            tgt, _ = self.cross_attn[i](tgt, kv=src, attn_bias=cross_mask, kv_cache={'k':None, 'v':None})
            
            # dec  # TODO: play with other attn_bias for decoder than for encoder
            tgt, kv_cache = self.decoder[i](tgt, attn_bias=attn_bias, kv_cache=kv_caches_dec[i])
            kv_caches_dec[i] = kv_cache

        src = self.enc_norm(src)
        enc_pred = self.enc2bbpe(src)
        enc_lens = src_lens

        tgt = self.dec_norm(tgt)
        dec_pred = self.dec2bbpe(tgt)

        return enc_pred, enc_lens, dec_pred, kv_caches_enc, kv_caches_dec
    

    
    def forward_simulate(self, mels_tensor, mel_lens, kv_caches=None):
        # enc emb
        mels_tensor = mels_tensor.unsqueeze(1)
        cnn_out, cnn_lens = self.cnnemb(mels_tensor, mel_lens)
        # enc
        #attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=self.window_size-1, window_right=0)
        # https://facebookresearch.github.io/xformers/components/ops.html
        attn_bias = fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(_window_size=self.window_size)
        if kv_caches is None: kv_caches = [{'k':None, 'v':None} for _ in range(len(self.encoder))]
        
        # simulate streaming input by looping over the sequence with random chunk sizes
        import random
        srcs = []
        cnn_pointer = 0
        while cnn_pointer < cnn_out.shape[1]:
            random_chunk_size = random.randint(1, 5)
            if cnn_pointer < self.window_size:
                random_chunk_size = self.window_size*2
            src = cnn_out[:, cnn_pointer:cnn_pointer+random_chunk_size]
            for i, layer in enumerate(self.encoder):
                src, kv_cache = layer(src, attn_bias, kv_cache=kv_caches[i])
                kv_caches[i] = kv_cache
            src = self.norm(src)
            srcs.append(src)
            cnn_pointer += random_chunk_size
        src = torch.cat(srcs, dim=1)
        enc_pred = self.enc2bbpe(src)
        return enc_pred, cnn_lens    