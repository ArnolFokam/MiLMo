import torch


def get_optimizer(model, train_cfg):
    if train_cfg.optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), 
            lr=train_cfg.lr, 
            weight_decay=train_cfg.weight_decay
        )
    elif train_cfg.optimizer_name == "sdg":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.lr, 
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay
        )
    else:
        raise ValueError(f"{train_cfg.optimizer_name} is not supported as an optimizer")
    

def get_lr_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler_name == "cosine": 
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=train_cfg.max_epochs, 
            eta_min=train_cfg.warmup_start_lr
        )
    else:
        return None