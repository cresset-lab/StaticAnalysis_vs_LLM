import time
import requests
import json
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro", requests_per_minute: int = 30):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
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
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
            }
        }
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = requests.post(self.url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                self.consecutive_api_errors = 0
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text']
                
                return ""
                
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
                    print(f"HTTP Error with Gemini API: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"Response: {e.response.text}")
                    return ""
                    
            except Exception as e:
                print(f"Error with Gemini API: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return ""
        
        print(f"Failed after {max_retries} retries")
        return ""
    
    def get_name(self) -> str:
        return f"gemini_{self.model.replace('-', '_').replace('.', '_')}"