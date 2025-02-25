import torch
import numpy as np
import os

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, D, split="train"):
        self.B = B
        self.T = T
        self.D = D
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = 0
    
    def next_batch(self):
        B, T , D = self.B, self.T, self.D
        buf = self.tokens[self.current_pos : self.current_pos + B * (T + D) + 1]
        x = buf[:-1].view(B, T + D)
        y = buf[1:].view(B, T + D)
        
        self.current_pos += B * (T + D)
        if self.current_pos + (B * (T + D) + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = 0
        return x, y