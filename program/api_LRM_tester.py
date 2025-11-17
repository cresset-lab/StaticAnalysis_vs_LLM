"""
Main entry point for running LLM experiments on openHAB rulesets
"""
from experiment_runner import LLMTester


def main():
    """Run the experiment workflow"""
    
    # Configuration
    config_path = 'config/api_keys.json'
    base_dir = r''
    dataset_path = 'mutated_dataset.csv'
    version = 'MUTATED_V1'
    
    # Initialize the tester
    print("Initializing LLM Tester...")
    tester = LLMTester(config_path=config_path)
    
    tester.setup_providers(provider_names=[
        'ollama-DS7b-Q4_K_M',
        #'gemini',
        # 'deepseek_siliconflow',
        # 'deepseek_native'
    ])
    
    # Run experiments
    print("\nStarting experiments...")
    tester.run_experiments(
        base_dir=base_dir,
        dataset_path=dataset_path,
        version=version,
        exclude_experiments=[]  # Add experiments to skip
    )
    
    print("\n✓ All experiments completed successfully!")


if __name__ == '__main__':
    main()