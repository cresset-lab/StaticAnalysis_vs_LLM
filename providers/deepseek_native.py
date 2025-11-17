import time
from openai import OpenAI
from .base import LLMProvider


class DeepSeekNativeProvider(LLMProvider):
    """Provider for DeepSeek using native DeepSeek API with OpenAI client"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", requests_per_minute: int = 30):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.requests_per_minute = requests_per_minute
        self.delay_between_requests = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.consecutive_api_errors = 0
        self.base_429_wait_time = 2.0
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_between_requests:
            time.sleep(self.delay_between_requests - time_since_last)
        self.last_request_time = time.time()
    
    def generate_response(self, prompt: str, max_retries: int = 4) -> str:
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    temperature=1.0
                )
                
                self.consecutive_api_errors = 0
                
                # Extract the response content
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    print(content)
                    return content if content else ""
                
                return ""
                
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit or service error
                if '429' in error_message or '503' in error_message:
                    self.consecutive_api_errors += 1
                    
                    if '429' in error_message:
                        error_type = "Rate limit hit"
                        backoff_base = 2
                    else:
                        error_type = "Service overloaded"
                        backoff_base = 2
                    
                    wait_time = self.base_429_wait_time * (backoff_base ** attempt)
                    print(f"{error_type}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error with DeepSeek Native API: {e}")
                    return ""
        
        print(f"Failed after {max_retries} retries")
        return ""
    
    def get_name(self) -> str:
        return f"deepseek_native_{self.model.replace('-', '_')}"