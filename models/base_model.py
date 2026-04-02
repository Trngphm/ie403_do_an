from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate(self, prompt):
        pass