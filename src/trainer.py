import os
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from .helpers import log, save_pytorch_things, time_activity
from .optimization import get_lr_scheduler, get_optimizer


class Trainer:
    
    TRAIN_PREFIX = "train"
    VALIDATION_PREFIX = "val"
    TESTING_PREFIX = "test"
    
    def __init__(self, train_cfg, results_dir) -> None:
        self.train_cfg = train_cfg
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_to_device(self, tensor):
        
        if isinstance(tensor, (List, Tuple)):
            for i in range(len(tensor)):
                tensor[i] = tensor[i].to(self.device, non_blocking=True)
        elif isinstance(tensor, (torch.Tensor, nn.Module)):
            tensor = tensor.to(self.device, non_blocking=True)

        return tensor
    
    def setup(self, model, data):
        # setup data
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.test_dataloader = data.test_dataloader()
        
        # setup model
        self.model = self.set_to_device(model)
        
        # Load model and optimzers from checkpoints or scratch
        if os.path.exists(os.path.join(self.results_dir, 'checkpoint.pth')):
            logging.info("Loading checkpoint from {}".format(os.path.join(self.results_dir, 'checkpoint.pth')))
            checkpoint = torch.load(os.path.join(self.results_dir, 'checkpoint.pth'))
            
            # model setup configuration
            # self.model =  torch.jit.script(model).to(self.device)
            self.model =  model.to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # optimizer configuration
            self.optimizer = get_optimizer(self.model, self.train_cfg)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler = get_lr_scheduler(self.optimizer, self.train_cfg)
            
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch'] + 1
            logging.info("Loaded checkpoint up till epoch {}".format(self.current_epoch))
        else:
            logging.info("No checkpoint found. Starting from scratch")
            # model setup configuration
            # self.model =  torch.jit.script(model).to(self.device)
            self.model =  model.to(self.device)
            self.model.eval()
            
            # optimizer and scheduler configuration
            self.optimizer = get_optimizer(self.model, self.train_cfg)
            self.scheduler = get_lr_scheduler(self.optimizer, self.train_cfg)
            
            self.current_epoch = 0
        self.global_step = max(0, self.current_epoch - 1) * len(self.train_dataloader)

    def fit_and_test(self, model, data):
        
        self.current_loss = 0
        
        # setup model, data and state
        self.setup(model, data)
        
        with time_activity("Training"):
            for _ in range(self.current_epoch, self.train_cfg.max_epochs):
                
                with time_activity("Epoch {}".format(self.current_epoch + 1)):
    
                    self.fit_epoch()
                    
                    if self.current_epoch % self.train_cfg.save_every_n_epochs == 0:
                        torch.save({
                            'epoch': self.current_epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                            }, os.path.join(self.results_dir,  'checkpoint.pth'))
            
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # per epoch logging
                    log(self.results_dir, {
                        f"epoch": self.current_epoch, 
                        "lr": self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step) 
                    
                        
                self.current_epoch += 1
                    
        # test model
        with time_activity("Testing"):
            self.test()
        
        # save model
        return save_pytorch_things(self.results_dir, {
            'model_state_dict': self.model.state_dict(),
            'cfg': self.train_cfg,
        })
    
    def fit_epoch(self):
        
        self.model.train()
        
        # training epoch
        total_loss, total_num = 0.0, 0
        for _, inputs in enumerate(self.train_dataloader, 1):
            
            # forward pass
            inputs = self.set_to_device(inputs)
            loss = self.model(inputs[:, :-1], inputs[:, 1:])["loss"]
            
            # backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            # record the loss
            self.current_loss = loss.item()
            
            # calculate running average of code
            total_num += self.train_cfg.batch_size
            total_loss += loss.item() * self.train_cfg.batch_size
            
            if self.global_step % self.train_cfg.log_every_n_steps == 0:
                log(self.results_dir, {"train-loss": total_loss / total_num}, step=self.global_step)
                
            # update the step
            self.global_step += 1 
                
        # do validation stuffs
        self.validate()

    
    def evaluate(self, mode):
        
        self.model.eval()
        
        dataloader = getattr(self, f"{mode}_dataloader", None)
        
        if dataloader:
            
            # training epoch
            total_num, tot_loss, tot_top1, tot_top5 = 0, 0.0, 0.0, 0.0
            for _, inputs in enumerate(self.train_dataloader, 1):
                
                with torch.no_grad():
                    
                    # forward pass
                    inputs = self.set_to_device(inputs)
                    outputs = self.model(inputs[:, :-1], inputs[:, 1:])

                total_num += (inputs.size(0) * inputs.size(1))
                predictions = torch.argsort(outputs["logits"], dim=-1, descending=True)
                tot_loss += outputs["loss"].item() * inputs.size(1)
                tot_top1 += torch.sum((predictions[:, :, 0:1] == inputs[:, 1:].unsqueeze(dim=-1)).any(dim=-1).float()).item()
                tot_top5 += torch.sum((predictions[:, :, 0:5] == inputs[:, 1:].unsqueeze(dim=-1)).any(dim=-1).float()).item()
            
            log(self.results_dir, {
                f"{mode}-loss": tot_loss / total_num,
                f"{mode}-top1-acc": tot_top1 / total_num * 100,
                f"{mode}-top5-acc": tot_top5 / total_num * 100,
            }, step=self.global_step) 
    
    def validate(self):
        """validation epoch"""
        self.evaluate(mode=self.VALIDATION_PREFIX)

    def test(self):
        """test epoch"""
        self.evaluate(mode=self.TESTING_PREFIX)