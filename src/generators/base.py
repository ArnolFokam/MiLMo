class BaseGenerator:
    def __init__(self, model, train_cfg) -> None:
        self.model = model
        self.train_cfg = train_cfg
    
    def generator(self):
        # generate `self.train_cfg.generated_num_blocks` num of blocks
        for _ in range(self.train_cfg.generation.num_blocks_per_generation):
            self.moo