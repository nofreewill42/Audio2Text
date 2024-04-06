

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
        self.layernormkv = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, kv=None, attn_bias=None, kv_cache=None):
        residual = x
        x = self.layernorm1(x)
        q = self.Q(x)
        q = q.reshape(*q.shape[:2], self.n_heads, -1)
        if kv is None:
            k, v = self.K(x), self.V(x)
            k = k.reshape(*k.shape[:2], self.n_heads, -1)
            v = v.reshape(*v.shape[:2], self.n_heads, -1)
        else:
            k, v = kv['k'], kv['v']

        # Input tensors must be in format [B, M, H, K] where
        # B is the batch size
        # M the sequence length
        # H the number of heads and
        # K the embeding size per head

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
    #def __init__(self, n_bbpe, n_layers=6, d_model=512, d_ff=2048, n_heads=8, window_size=48, dropout=0.0):
    def __init__(self, config):
        super().__init__()

        # CONFIG - START
        # "global" config
        n_bbpe = config["n_bbpe"]
        d_model = config["d_model"]
        # model config
        model_config = config["model"]
        d_ff = model_config["d_ff"]
        n_heads = model_config["n_heads"]
        n_layers = model_config["n_layers"]
        dropout = model_config["dropout"]
        # self
        self.n_bbpe, self.n_layers, self.d_model, self.d_ff, self.n_heads, self.dropout = n_bbpe, n_layers, d_model, d_ff, n_heads, dropout
        self.window_size = (model_config["window_size"][0], model_config["window_size"][-1]) # model_config["window_size"] is [window_left, 1, window_right]
        # CONFIG - END

        # Encoder embeddings
        self.cnnemb = CNNEmbedder(config)
        # Encoder
        self.encoder = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)
        self.enc2bbpe = nn.Linear(d_model, n_bbpe)

        # Decoder
        self.bbpe_emb = nn.Embedding(n_bbpe, d_model)
        self.cross_attn = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([XMHA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decnorm = nn.LayerNorm(d_model)
        self.dec2bbpe = nn.Linear(d_model, n_bbpe)
    
    def forward(self, bbpes_tensor, mels_tensor, mel_lens, kv_caches=None, n_cnn_processed=0):
        # enc emb
        mels_tensor = mels_tensor.unsqueeze(1)
        src, src_lens = self.cnnemb(mels_tensor, mel_lens, n_cnn_processed)
        # enc
        # https://facebookresearch.github.io/xformers/components/ops.html
        #attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=self.window_sizes[i][0], window_right=self.window_sizes[i][1])
        #attn_bias = fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(_window_size=self.window_size)
        if kv_caches is None: kv_caches = [{'k':None, 'v':None} for _ in range(len(self.encoder))]
        for i, layer in enumerate(self.encoder):
            #attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=63, window_right=0)
            attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=63, window_right=0)
            src, kv_cache = layer(src, attn_bias=attn_bias, kv_cache=kv_caches[i])
            kv_caches[i] = kv_cache
        src = self.norm(src)
        enc_pred = self.enc2bbpe(src)
        enc_lens = src_lens

        # dec
        tgt = self.bbpe_emb(bbpes_tensor)
        dec_kv_caches = [{'k':None, 'v':None} for _ in range(len(self.decoder))]
        for i, layer in enumerate(self.decoder):
            #attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=63, window_right=0)
            attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=3000, window_right=0)
            tgt, kv_cache = layer(tgt, attn_bias=attn_bias, kv_cache=dec_kv_caches[i])
            dec_kv_caches[i] = kv_cache
            tgt, _ = self.cross_attn[i](tgt, kv_caches[i], attn_bias=None, kv_cache={'k':None, 'v':None})
        tgt = self.decnorm(tgt)
        dec_pred = self.dec2bbpe(tgt)

        return dec_pred, enc_pred, enc_lens, kv_caches
    

    def encoder_forward(self, mels_tensor, mel_lens, kv_caches=None):
        # enc emb
        mels_tensor = mels_tensor.unsqueeze(1)
        src, src_lens = self.cnnemb(mels_tensor, mel_lens)
        # enc
        attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=3000, window_right=0)
        if kv_caches is None: kv_caches = [{'k':None, 'v':None} for _ in range(len(self.encoder))]
        for i, layer in enumerate(self.encoder):
            src, kv_cache = layer(src, attn_bias=attn_bias, kv_cache=kv_caches[i])
            kv_caches[i] = kv_cache
        src = self.norm(src)
        enc_pred = self.enc2bbpe(src)
        return enc_pred, src_lens, kv_caches
    
    def decoder_forward(self, bbpes_tensor, kv_caches=None, dec_kv_caches=None):
        tgt = self.bbpe_emb(bbpes_tensor)
        if dec_kv_caches is None: dec_kv_caches = [{'k':None, 'v':None} for _ in range(len(self.decoder))]
        for i, layer in enumerate(self.decoder):
            attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=3000, window_right=0)
            tgt, kv_cache = layer(tgt, attn_bias=attn_bias, kv_cache=dec_kv_caches[i])
            dec_kv_caches[i] = kv_cache
            tgt, _ = self.cross_attn[i](tgt, kv_caches[i], attn_bias=None, kv_cache={'k':None, 'v':None})
        tgt = self.decnorm(tgt)
        dec_pred = self.dec2bbpe(tgt)
        return dec_pred, dec_kv_caches

    

    
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