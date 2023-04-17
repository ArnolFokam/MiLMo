import math

import torch
from torch import nn, Tensor

class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10000.0))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape shape [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor: the tensor and the positional encoding added to it
        """
        
        assert len(x.shape) == 3, f"{self.__class__.__name__} receives a 3D tensor"
        return x + self.pe[:, :x.size(1), :]
    
if __name__ == "__main__":
    pe = PositionalEncoding1D(4, 2)
    print(pe(torch.randn((2, 4, 2))))