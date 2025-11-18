import os
import json
import pandas as pd
from typing import List, Optional
from providers import (
    LLMProvider, 
    GeminiProvider, 
    DeepSeekSiliconFlowProvider,
    DeepSeekNativeProvider,
    OllamaProvider,
    TMUProvider
)


class LLMTester:
    """Main class to handle testing across multiple providers"""
    
    def __init__(self, config_path: str = 'config/api_keys.json'):
        """
        Initialize the tester with API configurations
        
        Args:
            config_path: Path to JSON file with API keys and settings
        """
        self.providers = []
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                "Please create a config/api_keys.json file with your API keys."
            )
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def setup_providers(self, provider_names: Optional[List[str]] = None):
        """
        Setup providers based on configuration
        
        Args:
            provider_names: List of provider names to use (e.g., ['gemini', 'deepseek_native'])
                          If None, uses all available providers in config
        """
        if provider_names is None:
            provider_names = list(self.config.keys())
        
        for name in provider_names:
            if name not in self.config:
                print(f"Warning: Provider '{name}' not found in config, skipping...")
                continue
            
            provider_config = self.config[name]
            
            if name == 'gemini':
                self.providers.append(GeminiProvider(
                    api_key=provider_config['api_key'],
                    model=provider_config.get('model', 'gemini-2.5-pro'),
                    requests_per_minute=provider_config.get('requests_per_minute', 120)
                ))
            elif name == 'deepseek_siliconflow':
                self.providers.append(DeepSeekSiliconFlowProvider(
                    api_key=provider_config['api_key'],
                    model=provider_config.get('model', 'deepseek-ai/DeepSeek-R1'),
                    requests_per_minute=provider_config.get('requests_per_minute', 30)
                ))
            elif name == 'deepseek_native':
                self.providers.append(DeepSeekNativeProvider(
                    api_key=provider_config['api_key'],
                    model=provider_config.get('model', 'deepseek-chat'),
                    requests_per_minute=provider_config.get('requests_per_minute', )
                ))
            elif name == 'ollama-DS14b-Q4_K_M':
                self.providers.append(OllamaProvider(
                    model=provider_config.get('model','deepseek-r1:14b'),
                    base_url=provider_config.get('base_url', 'http://localhost:11434/v1')
                ))
            elif name == 'ollama-DS8b-Q4_K_M':
                self.providers.append(OllamaProvider(
                    model=provider_config.get('model','deepseek-r1:8b'),
                    base_url=provider_config.get('base_url', 'http://localhost:11434/v1')
                ))
            elif name == 'ollama-DS7b-Q4_K_M':
                self.providers.append(OllamaProvider(
                    model=provider_config.get('model','deepseek-r1:7b'),
                    base_url=provider_config.get('base_url', 'http://localhost:11434/v1')
                ))
            elif name == 'TMU_deepseek':
                self.providers.append(TMUProvider(
                    api_key=provider_config.get('api_key', 'ollama'),
                    base_url=provider_config.get('base_url', 'http://localhost:44014'),
                    model=provider_config.get('model', 'deepseek-r1:7b'),
                    requests_per_minute=provider_config.get('requests_per_minute', 120)
                ))
            elif name == 'TMU_llama_8b':
                self.providers.append(TMUProvider(
                    api_key=provider_config.get('api_key', 'ollama'),
                    base_url=provider_config.get('base_url', 'http://localhost:44014'),
                    model=provider_config.get('model', 'llama3.1:8b'),
                    requests_per_minute=provider_config.get('requests_per_minute', 120)
                ))
            elif name == 'TMU_llama_70b':
                self.providers.append(TMUProvider(
                    api_key=provider_config.get('api_key', 'ollama'),
                    base_url=provider_config.get('base_url', 'http://localhost:44014'),
                    model=provider_config.get('model', 'llama3.1:70b'),
                    requests_per_minute=provider_config.get('requests_per_minute', 300)
                ))
            else:
                print(f"Warning: Unknown provider type '{name}', skipping...")
        
        print(f"Loaded {len(self.providers)} provider(s): {[p.get_name() for p in self.providers]}")
    
    def apply_api(self, dataframe: pd.DataFrame, prompt: str, provider: LLMProvider, 
                  output_path: str, resume: bool = True) -> pd.DataFrame:
        """
        Apply API calls to classify rulesets and save after each response
        
        Args:
            dataframe: DataFrame containing rulesets
            prompt: The prompt template to use
            provider: The LLM provider to use
            output_path: Path where results should be saved
            resume: If True, skip rows that already have results
        
        Returns:
            DataFrame with LLM results
        """
        ruleset = list(dataframe['ruleset'])
        
        # Create a copy of the dataframe for results
        results_df = dataframe.copy()
        
        # Initialize LLM_result column if it doesn't exist
        if 'LLM_result' not in results_df.columns:
            results_df['LLM_result'] = ''
        
        # Determine which indices need processing
        indices_to_process = []
        
        if resume and os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                if 'LLM_result' in existing_df.columns:
                    # Copy existing results
                    results_df['LLM_result'] = existing_df['LLM_result']
                    
                    # Find ALL empty rows (not just the first one)
                    for i, result in enumerate(existing_df['LLM_result']):
                        if pd.isna(result) or result == '' or str(result).strip() == '':
                            indices_to_process.append(i)
                    
                    if indices_to_process:
                        filled_count = len(existing_df) - len(indices_to_process)
                        print(f"📂 Resuming: Found {filled_count} existing results, {len(indices_to_process)} empty rows to process")
                        print(f"   Empty row indices: {indices_to_process[:10]}{' ...' if len(indices_to_process) > 10 else ''}")
                    else:
                        print(f"✅ All {len(existing_df)} rows already have results - nothing to process!")
                        return results_df
                else:
                    # No LLM_result column, process all
                    indices_to_process = list(range(len(ruleset)))
            except Exception as e:
                print(f"⚠️ Could not read existing file, starting fresh: {e}")
                indices_to_process = list(range(len(ruleset)))
        else:
            # No existing file or not resuming, process all rows
            indices_to_process = list(range(len(ruleset)))
        
        # Prepare thinking process output path
        thinking_output_path = output_path.replace('.csv', '_Think_Process.txt')
        
        print(f"\nTesting with provider: {provider.get_name()}")
        print(f"Processing {len(indices_to_process)} rules out of {len(ruleset)} total")
        
        # Process only the indices that need work
        for progress_num, i in enumerate(indices_to_process, 1):
            rule = ruleset[i]
            full_prompt = prompt + rule + "\nCan we classify these openHAB rulesets as any of the rule interaction threats mentioned before?"
      
            print(f"[{progress_num}/{len(indices_to_process)}] Processing rule {i+1}/{len(ruleset)}...", end=" ", flush=True)
            
            # Retry logic for empty responses
            max_retries = 3
            response_data = None
            
            for attempt in range(max_retries):
                response_data = provider.generate_response(full_prompt)
                
                # Handle both dict responses (from Ollama) and string responses (from other providers)
                if isinstance(response_data, dict):
                    response = response_data.get('content', '')
                    thinking = response_data.get('thinking', '')
                else:
                    # Legacy string response from other providers
                    response = response_data if response_data else ""
                    thinking = ""
                
                # Check if response is valid (not empty, None, or just whitespace)
                if response and response.strip():
                    break  # Got valid response, exit retry loop
                else:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Empty (attempt {attempt + 1}/{max_retries}), retrying...", end=" ", flush=True)
                    else:
                        print(f"❌ Empty after {max_retries} attempts", end=" ", flush=True)
            
            # Save the response to the dataframe (even if empty after all retries)
            results_df.at[i, 'LLM_result'] = response if response else ""
            
            # Save thinking process if present
            if thinking:
                try:
                    with open(thinking_output_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Rule {i+1} - Thinking Process\n")
                        f.write(f"{'='*80}\n")
                        f.write(thinking)
                        f.write(f"\n{'='*80}\n\n")
                except Exception as e:
                    print(f"⚠️ Error saving thinking process: {e}", end=" ")
            
            # Save to CSV immediately after each response
            try:
                results_df.to_csv(output_path, index=False)
                if response and response.strip():
                    think_indicator = " (+ thinking)" if thinking else ""
                    print(f"✓ Saved{think_indicator}")
                else:
                    print("⚠️ Saved (empty)")
            except Exception as e:
                print(f"✗ Error saving: {e}")
        
        print(f"\n✅ Completed processing {len(indices_to_process)} rules")
        if os.path.exists(thinking_output_path):
            print(f"💭 Thinking processes saved to: {thinking_output_path}")
        
        return results_df
    
    def run_experiments(self, 
                       base_dir: str, 
                       dataset_path: str = 'dataset.csv',
                       version: str = 'VERSION3',
                       exclude_experiments: Optional[List[str]] = None,
                       resume: bool = True):
        """
        Run experiments across all providers and prompt templates
        
        Args:
            base_dir: Base directory containing experiment folders
            dataset_path: Path to the dataset CSV
            version: Version string for output files
            exclude_experiments: List of experiment folder names to exclude (e.g., ['experiment_e', 'experiment_a'])
            resume: If True, resume from existing results files instead of starting over
        """
        if exclude_experiments is None:
            exclude_experiments = ['experiment_e']
        
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} entries from {dataset_path}")
        experiment_count = 0
        
        for root, _, files in os.walk(base_dir):
            # Check if we should skip this directory
            should_skip = (
                'experiment' not in root or
                'results' in root or
                'venv' in root or
                'output' in root or
                any(excluded in root for excluded in exclude_experiments)
            )
            
            if should_skip:
                continue
            
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    
                    with open(file_path) as f:
                        prompt = f.read()
                    
                    for provider in self.providers:
                        print(f"\n{'='*50}")
                        print(f"Processing: {file_path}")
                        print(f"Provider: {provider.get_name()}")
                        print(f"{'='*50}")
                        
                        # Create results directory if it doesn't exist
                        results_dir = os.path.join(root, 'results', provider.get_name())
                        os.makedirs(results_dir, exist_ok=True)
                        
                        # Define output path
                        output_path = os.path.join(results_dir, f"{file}_{version}_RESULTS.csv")
                        
                        # Apply the API with incremental saving
                        results_df = self.apply_api(df, prompt, provider, output_path, resume=resume)
                        
                        experiment_count += 1
                        print(f'📊 Completed experiment {experiment_count}: {output_path}')
        
        print(f"\n{'='*50}")
        print(f"✅ Completed {experiment_count} experiments total")
        print(f"{'='*50}")
