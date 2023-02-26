"""
credits: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

from typing import Optional
import torch
from torch import Tensor
from torch.utils.data import dataset


def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw intergers into a flat Tensor."""
    data = [torch.tensor(item, dtype=torch.long) for item in raw_text_iter]
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