class BaseGenerator:
    def __init__(self, model, train_cfg) -> None:
        self.model = model
        self.train_cfg = train_cfg
    
    def generate(self):
        raise NotImplementedError