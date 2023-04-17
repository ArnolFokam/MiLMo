from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    """single-head attention layer"""
    
    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        
    def forward(self, x, mask):
        _, seq_len, dim = x.shape
        
        # [batch_size x seq_len x d_hidden] => [batch_size x seq_len x d_hidden]
        k = self.key(x)
        q = self.query(x)
        v = self.query(x)
        
        # attention
        weights = q @ k.transpose(-2, -1) * dim ** -0.5 # [batch_size x seq_len x d_hidden] => [batch_size x seq_len x seq_len]
        weights = weights.masked_fill(mask[:seq_len, :seq_len] == 0, float('-inf')) # prevent attention to unwanted tokens
        weights = F.softmax(weights, dim=-1)
        return weights @ v


class MultiHeadAttention(nn.Module):
    """multi-head attention layer"""
    
    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, d_model // num_heads) for _ in range(num_heads)
        ])
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask):
        attn_x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        return self.projection(attn_x)

    
class ResidualLayerNorm(nn.Module):
    """Residual forward-pass and layer normalization"""
    def __init__(self, d_model: int, forwad_block: nn.Module, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.forward_block = forwad_block
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        return self.layer_norm(x + self.dropout(self.forward_block(x)))

    
class FeedForward(nn.Module):
    """Feed-forward layer of a transformers"""
    
    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    """Non-look-ahead attention layer. NOTE: WITHOUT CROSS-ATTENTION"""
    def __init__(self, d_model: int, d_feedforward: int, max_seq_len: int, num_heads: Optional[int] = 1, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        
        # mask for self-attention with look-ahead
        self.register_buffer("mask", torch.tril(torch.ones((max_seq_len, max_seq_len))))
        
        # attention and layer norm
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model
        )
        self.attention_addlayernorm = ResidualLayerNorm(
            d_model=d_model,
            dropout=dropout,
            forwad_block=lambda x: self.multi_head_attention(x, self.mask)
        )
        
        # feed-forward and layer nodem
        self.feedforward = FeedForward(d_model, d_feedforward)
        self.feedforward_addlayernorm = ResidualLayerNorm(
            d_model=d_model,
            dropout=dropout,
            forwad_block=self.feedforward
        )
        
    def forward(self, x):
        x = self.attention_addlayernorm(x)
        x = self.feedforward_addlayernorm(x)
        return x
