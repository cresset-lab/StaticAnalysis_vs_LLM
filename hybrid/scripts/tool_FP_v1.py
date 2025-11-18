import pandas as pd
import os
import requests
import ast
import re
from openai import OpenAI

def get_rit_prompt(df, index, prompt_folder, num_shot):
    # Get list of ohit rit detections
    ohit_results_list = list(df["trad_result"])

    # Get ohit detection for the current rule-pair
    ohit_result = ohit_results_list[index]

    # Define the prompt file based on the type of ohit result
    prompt_file = prompt_folder + ohit_result.lstrip() + f'_FP_verify_prompt_{num_shot}.txt'

    # Open and read the prompt file
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt






def run_llm(rules='', prompt=''):    
    client = OpenAI(
    api_key="",
            )
       
    # Prepare the message for the ChatCompletion API
    messages = [
        {"role": "user", "content": prompt + rules},
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages
    )
    # print("response:", response)
    # print("mess", response.choices[0].message.content)
    return response.choices[0].message.content

def find_rules_in_files(path, dataframe, prompt_folder, num_shot):
    rules = list(dataframe["rules"])
    file_paths = list(dataframe["file_name"])
    
    llm_result_list = [] 
    for index, rule in enumerate(rules):
        # Get get prompt tailored to the RIT that ohit detected
        prompt = get_rit_prompt(dataframe, index, prompt_folder, num_shot)
        
        string_of_rules = ""
        rule = ast.literal_eval(rule)
        with open(os.path.join(path, file_paths[index]), 'r') as a_file_opened:
            string_of_rules = ""
            between_index = False
            for line in a_file_opened: 
                if rule[0] in line or rule[1] in line:
                    between_index = True 
                    string_of_rules += line.replace('\n', '\\n')
                    continue
                if between_index: 
                    if re.search(r'\bend\b', line): 
                        print(line)
                        between_index = False
                    string_of_rules += line.replace("\n", "\\n")
            print(f"Processing file index: {index}")
            response = run_llm(rules=string_of_rules, prompt=prompt)
            print(f"The response from llm is:\n{response}")
            lines = response.split("json")
            last_line = lines[-1] if lines else None 
            print(f"Last line: {last_line}")
            llm_result_list.append(response)
    return llm_result_list


def evaluate_results(df):
    # Clean the trad_result column by stripping extra whitespace
    df['trad_result'] = df['trad_result'].str.strip()
    
    # Define the expected trad_result labels and metrics to track
    labels = ['WAC', 'SAC', 'WTC', 'STC', 'WCC', 'SCC']
    metrics = ['TP', 'FN', 'TN', 'FP']
    
    # Initialize a dictionary to hold counts for each label
    results = {label: {metric: 0 for metric in metrics} for label in labels}
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        trad_label = row['trad_result']
        truth_actual = row['truth_actual']
        truth_experiment = row['truth_experiment']
        
        # Use .strip() to ensure no extra spaces
        trad_label = trad_label.strip()
        
        # If you encounter an unexpected label, add it to the dictionary.
        if trad_label not in results:
            results[trad_label] = {metric: 0 for metric in metrics}
        
        # Compare truth_actual and truth_experiment and update the counts accordingly:
        if truth_actual == 'TP' and truth_experiment == 'TP':
            results[trad_label]['TP'] += 1
        elif truth_actual == 'TP' and truth_experiment == 'FP':
            results[trad_label]['FN'] += 1
        elif truth_actual == 'FP' and truth_experiment == 'FP':
            results[trad_label]['TN'] += 1
        elif truth_actual == 'FP' and truth_experiment == 'TP':
            results[trad_label]['FP'] += 1

    # Neatly display the results in a table format
    print("Results Summary:")
    header = "{:<10} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8} {:>5} {:>10}".format("Label", "TP", "FN", "TN", "FP", "|", "Correct", "Inc", "Accuracy")
    print(header)
    print("-" * len(header))
    for label, counts in results.items():
        print("{:<10} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8} {:>5} {:>10}".format(label,
                                                      counts['TP'],
                                                      counts['FN'],
                                                      counts['TN'],
                                                      counts['FP'],
                                                      "|",
                                                      counts['TP'] + counts['TN'],
                                                      counts['FP'] + counts['FN'],
                                                      round((counts['TP'] + counts['TN']) / (counts['TP'] + counts['TN'] + counts['FP'] + counts['FN']), 2)))

            
if __name__ == "__main__":
    '''
    NOURA
    You should just need to edit:
        the prompt_folder path for your system
        run_llm() function to use llama
        output_csv use llama in the file name
    '''
    # Define the folder containing the prompt files.
    prompt_folder = "LLM-FP-verification/"

    # To iterate through and select zero, one, or two-shot prompt in get_rit_prompt() function
    experiment_type = ['1']#, '1', '2']

    path = ''

    # Iterate through each file in the prompt folder.
    for num_shot in experiment_type:
        # Read in the dataset.
        df = pd.read_csv('dataset_ohit_fp.csv')

        # Process the prompt with your custom function.
        llm_result_list = find_rules_in_files(path, df, prompt_folder, num_shot)

        # Add the result list to the DataFrame.
        df['truth_experiment'] = llm_result_list

        evaluate_results(df)

        # Generate an output CSV file name that includes the prompt file name.
        output_csv = f'gpt4o_fp_results_{num_shot}.csv'
        df.to_csv(output_csv, index=False)

        print(f"Created CSV: {output_csv}")

