import time
import requests
import json
from .base import LLMProvider


class DeepSeekSiliconFlowProvider(LLMProvider):
    """Provider for DeepSeek R1 model via SiliconFlow"""
    
    def __init__(self, api_key: str, model: str = "deepseek-ai/DeepSeek-R1", requests_per_minute: int = 30):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1"
        self.url = f"{self.base_url}/chat/completions"
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 16384,
            "stop": ["null"],
            "temperature": 1.0,
            "response_format": {"type": "text"}
        }
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = requests.post(self.url, headers=headers, json=payload, stream=True)
                
                if response.status_code == 200:
                    self.consecutive_api_errors = 0
                    
                    full_content = ""
                    reasoning_content = ""
                    first_reasoning_output = True
                    first_content_output = True
                    
                    for chunk in response.iter_lines():
                        if chunk:
                            chunk_str = chunk.decode('utf-8').strip()
                            
                            try:
                                if chunk_str.startswith('data:'):
                                    chunk_str = chunk_str[6:].strip()
                                
                                if chunk_str == "[DONE]":
                                    break
                                
                                chunk_json = json.loads(chunk_str)
                                if 'choices' in chunk_json and isinstance(chunk_json['choices'], list) and len(chunk_json['choices']) > 0:
                                    choice = chunk_json['choices'][0]
                                    delta = choice.get('delta', {})
                                    
                                    current_reasoning = delta.get('reasoning_content')
                                    current_content = delta.get('content')
                                    finish_reason = choice.get('finish_reason', None)
                                    
                                    if finish_reason is not None:
                                        print(f"\nFinish reason: {finish_reason}")
                                    
                                    if current_reasoning is not None:
                                        if first_reasoning_output:
                                            print("Reasoning process:")
                                            first_reasoning_output = False
                                        reasoning_content += current_reasoning
                                        print(current_reasoning, end='', flush=True)
                                    
                                    if current_content is not None:
                                        if first_content_output:
                                            print("\n\n==============================\nResult:")
                                            first_content_output = False
                                        full_content += current_content
                                        print(current_content, end='', flush=True)
                            
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}", flush=True)
                                continue
                    
                    if full_content:
                        return full_content
                    else:
                        return ""
                
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 503]:
                    self.consecutive_api_errors += 1
                    
                    if e.response.status_code == 429:
                        error_type = "Rate limit hit"
                        backoff_base = 2
                    else:
                        error_type = "Service overloaded"
                        backoff_base = 2
                    
                    wait_time = self.base_429_wait_time * (backoff_base ** attempt)
                    print(f"{error_type} ({e.response.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"HTTP Error with DeepSeek SiliconFlow API: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"Response: {e.response.text}")
                    return ""
                    
            except Exception as e:
                print(f"Error with DeepSeek SiliconFlow API: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return ""
        
        print(f"Failed after {max_retries} retries")
        return ""
    
    def get_name(self) -> str:
        return f"deepseek_siliconflow_{self.model.replace('/', '_').replace('-', '_')}"