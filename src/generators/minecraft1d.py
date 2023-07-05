import os
import logging
import random

import torch
import numpy as np
from src.generators.base import BaseGenerator
from src.helpers import generate_random_string, get_dir
from src.transforms.operations import VocabIDToText


class Mincraft1DGenerator(BaseGenerator):
    def generate(self, dataset, output_dir, vocab = None):
        
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
            to_tokens = VocabIDToText(vocab)
            sequence = np.asarray([to_tokens(seq.tolist()) for seq in sequence])
            prefix = np.asarray([to_tokens(seq.tolist()) for seq in prefix])
            
            # filter the sequence and prefix to replace <EOS>, <UNK> with default token
            sequence = map(sequence, lambda seq : [
                dataset.default_token  if token.startswith('<') and token.endswith('>') else token 
                for token in seq
            ])
            
            # change the tokens to a format that can be 
            # saved and loaded in the minecraft module
            sequence = dataset.to_world_format(sequence)
            prefix = dataset.to_world_format(prefix)
            random_string = generate_random_string(5)
            
            
            
            # save the sampled sequence
            np.save(os.path.join(output_dir_samples, f"{random_string}.npy"), sequence)
            
            # save the generated sequences
            np.save(os.path.join(output_dir_generation, f"{random_string}.npy"), prefix)
            
        logging.info("Generation complete")