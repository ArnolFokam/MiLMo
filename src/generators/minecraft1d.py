import os
import logging
import random

import torch
import numpy as np
from src.generators.base import BaseGenerator
from src.helpers import generate_random_string, get_dir


class Mincraft1DGenerator(BaseGenerator):
    def generate(self, dataset, output_dir, vocab=None):
        
        # put the generations in a sub-folder
        output_dir_samples = get_dir(os.path.join(output_dir, "samples"))
        output_dir_generation = get_dir(os.path.join(output_dir, "generations"))
        
        logging.info(f"Generation started")
        for _ in range(self.cfg.generation.num_generations):
                
            # generate random prefix and we should predict the rest
            sequence = random.choice(dataset).unsqueeze(0)
            prefix = sequence[:, :sequence.shape[1] // 2]
                
            # generate the number of blocks remaining
            for _ in range(sequence.shape[1] - prefix.shape[1]):
                # get the next token while keeping the 
                # same dimensions for easy concatenation
                next_token = self.model.generate(prefix)[None, :, -1]
                prefix = torch.cat([prefix, next_token], dim=1)

            # save the generated sequence
            assert sequence.shape == prefix.shape
            
            # map vocabulary to blocks
            # sequence = np.asarray([vocab.get_itos(seq.tolist()) for seq in sequence])
            # prefix = np.asarray([vocab.get_itos(seq.tolist()) for seq in prefix])
            
            sequence = sequence.reshape(sequence.shape[0], dataset.world_shape[1], dataset.world_shape[2])
            prefix = prefix.reshape(prefix.shape[0], dataset.world_shape[1], dataset.world_shape[2])
            random_string = generate_random_string(5)
            
            # save the sampled sequence
            np.save(os.path.join(output_dir_samples, f"{random_string}.npy"), sequence)
            
            # save the generated sequences
            np.save(os.path.join(output_dir_generation, f"{random_string}.npy"), prefix)
            
        logging.info("Generation complete")