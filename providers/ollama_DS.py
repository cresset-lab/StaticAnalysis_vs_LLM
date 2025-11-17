from openai import OpenAI
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Provider for local Ollama models using OpenAI-compatible API"""
    
    def __init__(self, model: str = "deepseek-r1:14b", base_url: str = "http://localhost:11434/v1"):
        """
        Initialize Ollama provider
        
        Args:
            model: The Ollama model to use (e.g., "llama2", "mistral", "codellama")
            base_url: The Ollama API endpoint (default: http://localhost:11434/v1)
        """
        self.model = model
        self.client = OpenAI(
            api_key="ollama",  # Ollama doesn't need an API key
            base_url=base_url
        )
    
    def generate_response(self, prompt: str, max_retries: int = 9) -> dict:
        """
        Generate a response from Ollama
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Number of retries on failure (default: 9)
        
        Returns:
            Dictionary with 'content' (main response) and 'thinking' (thinking process if present)
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    temperature=1.0
                )
                
                # Extract the response content
                if response.choices and len(response.choices) > 0:
                    full_content = response.choices[0].message.content
                    
                    if not full_content:
                        return {"content": "", "thinking": ""}
                    
                    # Check if </think> tag is present (handles both complete and incomplete tags)
                    if '</think>' in full_content:
                        # Split on the closing tag
                        parts = full_content.split('</think>', 1)
                        
                        # Everything before </think> is thinking (remove any opening <think> tag if present)
                        thinking_process = parts[0].replace('<think>', '', 1).strip()
                        
                        # Everything after </think> is the actual content
                        clean_content = parts[1].strip() if len(parts) > 1 else ""
                        
                        return {"content": clean_content, "thinking": thinking_process}
                    else:
                        # No thinking tags present
                        return {"content": full_content, "thinking": ""}
                
                return {"content": "", "thinking": ""}
                
            except Exception as e:
                error_message = str(e)
                print(f"Error with Ollama API (attempt {attempt + 1}/{max_retries}): {error_message}")
                
                # Only retry if there are attempts left
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
                else:
                    return {"content": "", "thinking": ""}
        
        return {"content": "", "thinking": ""}
    
    def get_name(self) -> str:
        """Return the provider name for file naming"""
        return f"ollama_{self.model.replace('-', '_').replace(':', '_')}"