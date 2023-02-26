"""
credits: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

from typing import Any, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import dataset
import torchtext

# TODO: make this configurable
bptt = 35

# TODO: code minecraft supported tokenization
def preprocess_data(raw_text_iter: dataset.IterableDataset, vocab: torchtext.vocab.Vocab) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(item), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, bsz: int, device: Optional[str] = "cpu") -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

