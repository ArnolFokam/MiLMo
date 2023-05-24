import os
import logging
import random

import torch
import numpy as np
from src.generators.base import BaseGenerator
from src.helpers import generate_random_string, get_dir


class Mincraft1DGenerator(BaseGenerator):
    def generate(self, dataset, output_dir):
        
        # put the generations in a sub-folder
        output_dir_samples = get_dir(os.path.join(output_dir, "samples"))
        output_dir_generation = get_dir(os.path.join(output_dir, "generations"))
        
        logging.info(f"Generation started")
        for _ in range(self.train_cfg.generation.num_generations):
                
            # generate random prefix and we should predict the rest
            sequence = random.choice(dataset).unsqueeze(0)
            prefix = sequence[:, :sequence.shape[1] // 2]
                
            # generate `num_blocks` blocks from the prompt
            for _ in range(prefix.shape[1] // 2):
                predictions = self.model.generate(prefix)
                print(predictions.shape, prefix.shape)
                prefix = torch.cat([prefix, predictions], dim=1)

            # save the generated sequence
            sequence = sequence.reshape(sequence.shape[0], dataset.shape[1], dataset.shape[2])
            prefix = prefix.reshape(prefix.shape[0], dataset.shape[1], dataset.shape[2])
            random_string = generate_random_string(5)
            
            # save the sampled sequence
            np.save(os.path.join(output_dir_samples, f"{random_string}.npy"), sequence.detach().cpu().numpy())
            
            # save the generated sequences
            np.save(os.path.join(output_dir_generation, f"{random_string}.npy"), prefix.detach().cpu().numpy())
            
        logging.info("Generation complete")