import os
import glob
from typing import Any, Optional

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from src.transforms import RESERVED_TOKENS, UNK, pipelines

class test:
    data_dir='data/worlds/shmar'
    val_split=0.8
    seed=1
    batch_size=26
    max_seq_len=16
    min_crop_len=4
    

class MinecraftLanguageModelling1D(Dataset):
    def __init__(self, root: str, transform: Optional[callable] = None):
        self.root = root
        self.transform = transform
        
        # load all the blocks. we use character level tokenization
        self.world = np.load(self.root).astype('str')
        self.world_shape = self.world.shape
        self.world = self.world.reshape(self.world.shape[0], -1)
        
    def __len__(self):
        return self.world.shape[0]

    def __getitem__(self, idx) -> Any:
        blocks = self.world[idx]
        
        # build the transform
        if self.transform:
            blocks = self.transform(blocks)
            
        return blocks
    
class MinecraftDataModule1D:
    
    def __init__(self, cfg, transform_name) -> None:
        self.cfg = cfg
        
        # real dataset
        self.dataset = MinecraftLanguageModelling1D(root=self.cfg.data_dir)
        
        # build vocabularity
        self.vocab = build_vocab_from_iterator(self.dataset.world, specials=RESERVED_TOKENS, special_first=True)
        self.vocab.set_default_index(self.vocab.lookup_indices([UNK])[0])
        
        # number of tokens
        self.num_tokens = len(self.vocab)
        
        # create transforms
        self.dataset.transform = pipelines(self.cfg, self.vocab)[transform_name]
        
    def train_dataloader(self):
        """Loads the training dataloader"""
        data_len = len(self.dataset)
        val_len = int(data_len * self.cfg.val_split)
        dataset_train, _ = random_split(
            self.dataset,
            [data_len - val_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.device.seed),
        )
        
        loader = DataLoader(
            dataset_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(12, int(0.5 * os.cpu_count()))
        )
        return loader
        
    def val_dataloader(self):
        """Loads the training dataloader"""
        data_len = len(self.dataset)
        val_len = int(data_len * self.cfg.val_split)
        _, dataset_val = random_split(
            self.dataset,
            [data_len - val_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.device.seed),
        )
        
        loader = DataLoader(
            dataset_val,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(12, int(0.5 * os.cpu_count()))
        )
        return loader
    
    def test_dataloader(self):
        return None


