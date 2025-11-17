from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name for file naming"""
        pass