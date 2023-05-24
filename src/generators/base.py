class BaseGenerator:
    def __init__(self, model, cfg) -> None:
        self.model = model
        self.cfg = cfg
    
    def generate(self):
        raise NotImplementedError