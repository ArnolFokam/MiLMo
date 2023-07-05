from typing import Any, Callable, List

import torch
import torchtext


class Compose:
    """Compose transforms to build a pipeline"""
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms
        
    def __call__(self, tensor: torch.Tensor) -> Any:
        
        # pipeline for the transform
        for t in self.transforms:
            tensor = t(tensor)
            
        return tensor
    
class RandomCropPad:
    def __init__(self, min_crop_len: int, max_seq_len: int, padding_token: int) -> None:
        assert min_crop_len < max_seq_len
        self.max_seq_len = max_seq_len
        self.min_crop_len = min_crop_len
        self.padding_token = padding_token
    
    def __call__(self, inputs) -> Any:
        # note we add the +1 to the length for casuall language modelling
        start = torch.randint(0, inputs.shape[0] - self.max_seq_len - 1, (1,))
        length = torch.randint(self.min_crop_len, self.max_seq_len + 1, (1,))
        paddings = torch.full((self.max_seq_len - length + 1,), self.padding_token)
        return torch.cat([paddings, inputs[start:start + length]])
    
class TextToVocabID:
    def __init__(self, vocab: torchtext.vocab.Vocab) -> None:
        self.vocab = vocab
        
    def __call__(self, inputs) -> Any:
        return torch.tensor(self.vocab(list(inputs)))
    
class VocabIDToText:
    def __init__(self, vocab: torchtext.vocab.Vocab) -> None:
        self.vocab = vocab
        
    def __call__(self, inputs) -> Any:
        return self.vocab.lookup_tokens(inputs)
        