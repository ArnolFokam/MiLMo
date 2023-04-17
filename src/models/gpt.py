from typing import Optional
from torch import nn, Tensor
import torch
import torch.nn.functional as F

from src.models.layers import DecoderLayer
from src.models.positional_encoding import PositionalEncoding1D


class Test:
    ntoken = 20
    d_model = 128
    d_feedforward = 2048
    nhead = 3
    num_layers = 5
    batch_size = 10
    max_seq_len = 10
    vocab_size=50
    num_heads=16
    dropout=0.1


class GPT(nn.Module):
    def __init__(self, train_cfg, vocab_size: int):
        super().__init__()
        
        if train_cfg.d_model % train_cfg.num_heads != 0:
            raise RuntimeError(f"The hidden dimenision {train_cfg.d_model} must be divisible by the number of heads {train_cfg.num_heads}")

        self.train_cfg = train_cfg
        
        # tokens encoding
        self.token_embeddings = nn.Embedding(vocab_size,
                                             self.train_cfg.d_model)
        
        # positional encoding
        self.positional_encoder = PositionalEncoding1D(
            d_model=self.train_cfg.d_model, 
            max_seq_len=self.train_cfg.max_seq_len
        )
        
        # dropout
        self.dropout = nn.Dropout(p=self.train_cfg.dropout)
        
        # decoder layer of a transformer
        self.decoder = nn.Sequential(*[
            DecoderLayer(
                d_model=self.train_cfg.d_model,
                d_feedforward=self.train_cfg.d_feedforward,
                max_seq_len=self.train_cfg.max_seq_len,
                num_heads=self.train_cfg.num_heads,
                dropout=self.train_cfg.dropout
            )
            for _ in range(self.train_cfg.num_layers)
        ])
        
        # linear layer for classification
        self.classifier = nn.Linear(self.train_cfg.d_model, vocab_size)

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        # transformer attention
        x = self.token_embeddings(inputs)  # [batch_size x seq_len x d_model]
        x = self.positional_encoder(x)  # [batch_size x seq_len x d_model]
        x = self.decoder(self.dropout(x))  # [batch_size x seq_len x d_model]
        logits = self.classifier(x)

        # calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.contiguous().view(-1))

        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    model = GPT(Test, Test.vocab_size)
    inputs = torch.randint(0, Test.vocab_size, (Test.batch_size, Test.max_seq_len))
    targets = torch.randint(0, Test.vocab_size, (Test.batch_size, Test.max_seq_len))
    print(model(inputs, targets))