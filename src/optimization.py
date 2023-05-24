import torch


def get_optimizer(model, cfg):
    if cfg.optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer_name == "sdg":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr, 
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"{cfg.optimizer_name} is not supported as an optimizer")
    

def get_lr_scheduler(optimizer, cfg):
    if cfg.lr_scheduler_name == "cosine": 
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.max_epochs, 
            eta_min=cfg.warmup_start_lr
        )
    else:
        return None