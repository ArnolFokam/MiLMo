import glob
import os

import numpy as np
from torch.utils.data import Dataset


class MinecraftDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.blocks = glob.glob(os.path.join(root_dir, "*.npy"))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
       blocks = np.load(os.path.join(self.root_dir, self.blocks[idx])).flatten()
       return [str(i) for i in blocks]
   
if __name__ == "__main__":
    dataset = MinecraftDataset("/home/arnol/research/milt/data/worlds/shmar")
    print(len(dataset))
    print(dataset[0].shape)
    print(dataset[0])