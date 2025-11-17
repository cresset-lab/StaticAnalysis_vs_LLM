import time
import requests
import json
import re
from .base import LLMProvider


class TMUProvider(LLMProvider):
    """Provider for DeepSeek R1 running in Ollama Docker container"""
    
    def __init__(self, api_key: str = "ollama", base_url: str = "http://localhost:44014", 
                 model: str = "deepseek-r1:7b", requests_per_minute: int = 30):
        self.api_key = api_key  # Not used but kept for consistency
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.requests_per_minute = requests_per_minute
        self.delay_between_requests = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0
        self.consecutive_api_errors = 0
        self.base_429_wait_time = 2.0
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limit"""
        if self.delay_between_requests > 0:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay_between_requests:
                time.sleep(self.delay_between_requests - time_since_last)
            self.last_request_time = time.time()
    
    def _extract_thinking_and_content(self, full_response: str) -> dict:
        """
        Extract thinking process and final answer from DeepSeek R1 response.
        DeepSeek R1 models output their reasoning in <think> tags.
        
        Args:
            full_response: The complete model response
            
        Returns:
            Dictionary with 'thinking' and 'content' keys
        """
        # Pattern to match <think>...</think> tags
        think_pattern = r'<think>(.*?)</think>'
        
        # Find all thinking blocks (there might be multiple)
        thinking_matches = re.findall(think_pattern, full_response, re.DOTALL)
        
        if thinking_matches:
            # Combine all thinking blocks
            thinking = '\n\n'.join(match.strip() for match in thinking_matches)
            
            # Remove thinking blocks from response to get final answer
            content = re.sub(think_pattern, '', full_response, flags=re.DOTALL).strip()
        else:
            # No thinking tags found - treat entire response as content
            thinking = ""
            content = full_response.strip()
        
        return {
            'thinking': thinking,
            'content': content
        }
    
    def generate_response(self, prompt: str, max_retries: int = 4) -> dict:
        """
        Generate response from Ollama API with thinking process extraction.
        
        Args:
            prompt: The input prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with 'thinking' and 'content' keys, or empty dict on error
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                # Prepare the request payload for Ollama
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,  # Enable streaming to get tokens as they come
                    "options": {
                        "num_ctx": 8192,  # Context window size
                        "temperature": 0.2,
                        "top_p": 0.9
                    }
                }
                
                # Make the API request with streaming
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=300  # 5 minute timeout for long responses
                )
                
                response.raise_for_status()
                
                # Collect the streaming response
                full_response = ""
                
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            
                            # Append the response token
                            if 'response' in json_response:
                                full_response += json_response['response']
                            
                            # Check if generation is complete
                            if json_response.get('done', False):
                                break
                                
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON line: {e}")
                            continue
                
                self.consecutive_api_errors = 0
                
                # Extract thinking and content
                result = self._extract_thinking_and_content(full_response)
                
                # Print the final answer (not the thinking)
                if result['content']:
                    print(result['content'])
                
                # Return empty dict if no content
                if not result['content'] and not result['thinking']:
                    print("Warning: Empty response received")
                    return {'thinking': '', 'content': ''}
                
                return result
                
            except requests.exceptions.Timeout:
                print(f"Request timeout. Retry {attempt + 1}/{max_retries}...")
                time.sleep(self.base_429_wait_time * (2 ** attempt))
                continue
                
            except requests.exceptions.RequestException as e:
                error_message = str(e)
                
                # Check for rate limit or service errors
                if '429' in error_message or '503' in error_message:
                    self.consecutive_api_errors += 1
                    
                    error_type = "Rate limit hit" if '429' in error_message else "Service overloaded"
                    backoff_base = 2
                    
                    wait_time = self.base_429_wait_time * (backoff_base ** attempt)
                    print(f"{error_type}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error with Ollama API: {e}")
                    return {'thinking': '', 'content': ''}
                    
            except Exception as e:
                print(f"Unexpected error: {e}")
                return {'thinking': '', 'content': ''}
        
        print(f"Failed after {max_retries} retries")
        return {'thinking': '', 'content': ''}
    
    def get_name(self) -> str:
        """Return provider name for file naming"""
        # Clean up model name for filesystem
        model_clean = self.model.replace(':', '_').replace('-', '_')
        return f"TMU_{model_clean}"
