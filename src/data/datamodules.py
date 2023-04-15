import os
import glob
from typing import Any, Optional

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from src.data.transforms import Compose, RandomCropPad, ToVocabID

PAD = '<pad>'
EOB = '<EOS>'
SOB = '<SOS>'

RESERVED_TOKENS=[PAD, EOB, SOB]

class test:
    data_dir='data/worlds/shmar'
    val_split=0.8
    seed=1
    batch_size=26
    max_seq_len=16
    min_crop_len=4
    

class MinecraftLanguageModelling1D(Dataset):
    def __init__(self, root: str, transform: Optional[callable] = None, target_transform: Optional[callable] = None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        # load all the blocks. we use character level tokenization
        self.blocks = [np.load(block).flatten().astype('str') for block in glob.glob(os.path.join(root, "*.npy"))]
        
    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx) -> Any:
        blocks = self.blocks[idx]
        
        # build the transform
        if self.transform:
            blocks = self.transform(blocks)
            
        return blocks
    
class MinecraftDataModule1D:
    
    def __init__(self, train_cfg) -> None:
        self.train_cfg = train_cfg
        
        # real dataset
        self.dataset = MinecraftLanguageModelling1D(root=self.train_cfg.data_dir)
        
        # build vocabularity
        self.vocab = build_vocab_from_iterator(self.dataset.blocks, specials=RESERVED_TOKENS, special_first=True)
        
        # number of tokens
        self.num_tokens = len(self.vocab)
        
        # create transforms
        self.dataset.transform = Compose([
            ToVocabID(self.vocab),
            RandomCropPad(
                min_crop_len=self.train_cfg.min_crop_len,
                max_seq_len=self.train_cfg.max_seq_len,
                padding_token=self.vocab.lookup_indices([PAD])[0]
            )
        ])
        
    def train_dataloader(self):
        """Loads the training dataloader"""
        data_len = len(self.dataset)
        val_len = int(data_len * self.train_cfg.val_split)
        dataset_train, _ = random_split(
            self.dataset,
            [data_len - val_len, val_len],
            generator=torch.Generator().manual_seed(self.train_cfg.seed),
        )
        
        loader = DataLoader(
            dataset_train,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(12, int(0.5 * os.cpu_count()))
        )
        return loader
        
    def val_dataloader(self):
        """Loads the training dataloader"""
        data_len = len(self.dataset)
        val_len = int(data_len * self.train_cfg.val_split)
        _, dataset_val = random_split(
            self.dataset,
            [data_len - val_len, val_len],
            generator=torch.Generator().manual_seed(self.train_cfg.seed),
        )
        
        loader = DataLoader(
            dataset_val,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(12, int(0.5 * os.cpu_count()))
        )
        return loader


datasets = {
    "1d_text_blocks": MinecraftDataModule1D
}