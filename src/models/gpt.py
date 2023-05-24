from typing import Optional
from torch import nn, Tensor
import torch
import torch.nn.functional as F

from src.models.layers import DecoderLayer
from src.models.positional_encoding import PositionalEncoding1D


class GPT(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        
        if cfg.d_model % cfg.num_heads != 0:
            raise RuntimeError(f"The hidden dimenision {cfg.d_model} must be divisible by the number of heads {cfg.num_heads}")

        self.cfg = cfg
        
        # tokens encoding
        self.token_embeddings = nn.Embedding(vocab_size,
                                             self.cfg.d_model)
        
        # positional encoding
        self.positional_encoder = PositionalEncoding1D(
            d_model=self.cfg.d_model, 
            max_seq_len=self.cfg.max_seq_len
        )
        
        # dropout
        self.dropout = nn.Dropout(p=self.cfg.dropout)
        
        # decoder layer of a transformer
        self.decoder = nn.Sequential(*[
            DecoderLayer(
                d_model=self.cfg.d_model,
                d_feedforward=self.cfg.d_feedforward,
                max_seq_len=self.cfg.max_seq_len,
                num_heads=self.cfg.num_heads,
                dropout=self.cfg.dropout
            )
            for _ in range(self.cfg.num_layers)
        ])
        
        # linear layer for classification
        self.classifier = nn.Linear(self.cfg.d_model, vocab_size)

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
    
    @torch.no_grad()
    def generate(self, prefix: Tensor):
        
        # get the next block prediction
        logits = self(prefix[:, -self.cfg.max_seq_len:])["logits"]
        
        # focus only on the last token
        logits = logits[:, -1, :]
        
        # apply softmac to get the probabilities
        probs = F.softmax(logits, dim=-1)
        
        # sample the next token
        next_block = torch.multinomial(probs, num_samples=1)
        
        # append the sampled index to the running sequence
        outputs = torch.cat([prefix, next_block], dim=-1)
            
        return outputs
            


if __name__ == "__main__":
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
        generated_num_blocks=5
    
    model = GPT(Test, Test.vocab_size)
    inputs = torch.randint(0, Test.vocab_size, (Test.batch_size, Test.max_seq_len))
    targets = torch.randint(0, Test.vocab_size, (Test.batch_size, Test.max_seq_len))
    # print(model(inputs, targets))
    # print(inputs)
    # print(model.generate(inputs))