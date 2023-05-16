import sys
import os
import torch
import random
import numpy as np

import hydra
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.models import models
from src.data import datamodules
from src.trainer import Trainer
from src.generators import generators
from src.helpers import generate_random_string, get_dir, get_new_run_dir_params, has_valid_hydra_dir_params, initialize_logging

def train(
    output_dir: str,
    train_cfg,
) -> str:

    # compute some variables
    save_dir = get_dir(output_dir)

    # setup the data
    data = datamodules[train_cfg.dataset_name](train_cfg)
    
    # initialize models with petrained weights if available
    model = models[train_cfg.model_name](train_cfg, len(data.vocab))
    
    # initialize trainer
    trainer = Trainer(
        train_cfg,
        results_dir=save_dir
    )
    
    return trainer.fit_and_test(model=model, data=data)

def generate(
        train_cfg,
        pretrained_model_path: str,
    ):
    # load the pretrained model
    pretrained_model = torch.load(pretrained_model_path)
    model = models[train_cfg.model_name](train_cfg, pretrained_model["vocab_len"])
    model.load_state_dict(pretrained_model["model_state_dict"], strict=False)
    prefix = datamodules[train_cfg.dataset_name].prefix
    
    for _ in range(train_cfg.generation.num_generations):
        generator = generators[train_cfg.generation.generator_name](
            train_cfg=train_cfg,
            moddel=model
        )
        output = generator.generate(
            prefix=prefix, 
            num_blocks=train_cfg.generation.num_blocks_per_generation
            )
        np.save(os.path.join(train_cfg.output_dir, f"generated_{generate_random_string(5)}.npy"), output)

@hydra.main(version_base=None, config_path=None)
def main(cfg) -> None:
    """Main script to pretrain/finetune SSL algorithms"""
    
    # ensure reprodcibility and speed up
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    output_dir = HydraConfig.get().runtime.output_dir
    
    # setup logging thing
    # not the one use for metrics logging
    initialize_logging(os.path.join(output_dir, 'main.log'))
    
    # save configuration as json
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)
       
    # do language modelling
    pretrained_model_path = train(
        output_dir=output_dir,
        train_cfg=cfg,
    )
    
    # gerneate results
    generate(pretrained_model_path)
        
    # save the completion of state of the run
    open(os.path.join(output_dir, 'completed'), 'w').close()
    

if __name__ == "__main__":
    if has_valid_hydra_dir_params(sys.argv):
        main()
    else:
        params = get_new_run_dir_params()
        for param, value in params.items():
            sys.argv.append(f"{param}={value}")
        main()