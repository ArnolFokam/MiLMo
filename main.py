import logging
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
    cfg,
) -> str:

    # compute some variables
    save_dir = get_dir(output_dir)

    # setup the data
    data = datamodules[cfg.dataset_name](
        cfg, 
        cfg.transform_name
    )
    
    # initialize models with petrained weights if available
    model = models[cfg.model_name](cfg, len(data.vocab))
    
    # initialize trainer
    trainer = Trainer(
        cfg,
        results_dir=save_dir
    )
    
    return trainer.fit_and_test(model=model, data=data)

def generate(
        cfg,
        output_dir,
        pretrained_model_path: str,
    ):
    
    # load the pretrained model
    pretrained_model = torch.load(pretrained_model_path)
    model = models[cfg.model_name](cfg, pretrained_model["vocab_len"])
    model.load_state_dict(pretrained_model["model_state_dict"], strict=False)
    
    # get the data used during training
    dataset = datamodules[cfg.dataset_name](
        cfg,
        cfg.generation.transform_name
    ).dataset
    
    # load the generator
    generator = generators[cfg.generation.generator_name](
        cfg=cfg,
        model=model
    )
    
    # generate the blocks
    generator.generate(dataset=dataset, output_dir=output_dir, vocab=dataset.vocab)
        
        
        
@hydra.main(version_base=None, config_path=None)
def main(cfg) -> None:
    """Main script to pretrain/finetune SSL algorithms"""
    
    # ensure reprodcibility and speed up
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.device.seed)
        
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.device.seed)
    random.seed(cfg.device.seed)
    np.random.seed(cfg.device.seed)
    
    output_dir = HydraConfig.get().runtime.output_dir
    
    # setup logging thing
    # not the one use for metrics logging
    initialize_logging(os.path.join(output_dir, 'main.log'))
    
    # save configuration as json
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)
        
    # pretrained model
    pretrained_model_path = None
       
    if cfg.experiment.do_pretraining:
        # do language modelling
        pretrained_model_path = train(
            output_dir=output_dir,
            cfg=cfg,
        )
    
    if cfg.experiment.do_generation:
        # generate results with pretrained model if exists
        pretrained_model_path = cfg.generation.pretrained_model_path if pretrained_model_path is None else pretrained_model_path
        generate(
            cfg=cfg,
            output_dir=output_dir,
            pretrained_model_path=pretrained_model_path,
        )
        
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