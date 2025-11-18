import pandas as pd
import os
import requests
import ast
import re

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
    print("oHIT + LLM Results Summary:")
    header = "{:<10} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8} {:>5} {:>10}".format("Label", "TP", "FN", "TN", "FP", "|", "Correct", "Inc", "Accuracy")
    print(header)
    print("-" * len(header))
    overleaf_string = ''
    overleaf_total = 0
    for label, counts in results.items():
        print("{:<10} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8} {:>5} {:>10}".format(label,
                                                      counts['TP'],
                                                      counts['FN'],
                                                      counts['TN'],
                                                      counts['FP'],
                                                      "|",
                                                      counts['TP'] + counts['TN'],
                                                      counts['FP'] + counts['FN'],
                                                      round((counts['TP'] + counts['TN']) / (counts['TP'] + counts['TN'] + counts['FP'] + counts['FN']), 4)))
        percentage = 100 * ((counts['TP'] + counts['TN']) /
                    (counts['TP'] + counts['TN'] + counts['FP'] + counts['FN']))

        overleaf_string += f"{percentage:.2f}\\% & "

        overleaf_total += round((counts['TP'] + counts['TN']) / (counts['TP'] + counts['TN'] + counts['FP'] + counts['FN']), 4)
    print(overleaf_string, round(100 * overleaf_total/6,2))
        
def evaluate_ohit(df):
    # Clean the trad_result column by stripping extra whitespace
    df['trad_result'] = df['trad_result'].str.strip()
    
    # Define the expected trad_result labels and metrics to track (only TP and FP)
    labels = ['WAC', 'SAC', 'WTC', 'STC', 'WCC', 'SCC']
    metrics = ['TP', 'FP']
    
    # Initialize a dictionary to hold counts for each label
    results = {label: {metric: 0 for metric in metrics} for label in labels}
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        trad_label = row['trad_result'].strip()
        truth_actual = row['truth_actual']
        
        # If an unexpected label is encountered, add it to the dictionary.
        if trad_label not in results:
            results[trad_label] = {metric: 0 for metric in metrics}
        
        # Count truth_actual (which will be either 'TP' or 'FP')
        if truth_actual == 'TP':
            results[trad_label]['TP'] += 1
        elif truth_actual == 'FP':
            results[trad_label]['FP'] += 1

    # Display the results in a neatly formatted table
    print("\noHIT Results Summary:")
    header = "{:<10} {:>5} {:>5} {:>5} {:>10}".format("Label", "TP", "FP", "Total", "Accuracy")
    print(header)
    print("-" * len(header))
    
    for label, counts in results.items():
        total = counts['TP'] + counts['FP']
        accuracy = round(counts['TP'] / total, 4) if total > 0 else 0
        print("{:<10} {:>5} {:>5} {:>5} {:>10}".format(label,
                                                        counts['TP'],
                                                        counts['FP'],
                                                        total,
                                                        accuracy))
        
    
import pandas as pd

def fix_truth_experiment(input_csv, output_csv):
    """
    Loads a CSV, modifies truth_experiment for WCC/SCC rows, saves result.
    """

    # Load file
    df = pd.read_csv(input_csv)

    # Condition: trad_result is WCC or SCC
    mask = df['trad_result'].isin(['WCC', 'SCC'])

    # Update truth_experiment to TP for those rows
    df.loc[mask, 'truth_experiment'] = 'TP'

    # Save to new file
    df.to_csv(output_csv, index=False)

    print(f"Finished. Saved updated file to: {output_csv}")



if __name__ == "__main__":
    
    results_dataset_csv = r"C:\\Users\\jason\\OneDrive\Documents\\TMU\\CPS40A\\LLM-v-Static-Noura\\noura-openai\\LLM-FP-verification\\results\\NEW_gem_25_results_2.csv"
    results_dataset_csv2 = r"C:\\Users\\jason\\OneDrive\Documents\\TMU\\CPS40A\\LLM-v-Static-Noura\\noura-openai\\LLM-FP-verification\\results\\NEWnew_gem_25_results_2.csv"
    ohit_dataset_csv = r"C:\\Users\\jason\\OneDrive\Documents\\TMU\\CPS40A\\LLM-v-Static-Noura\\noura-openai\\LLM-FP-verification\\datasets_fp\\dataset_ohit_fp.csv"
    
    fix_truth_experiment(results_dataset_csv, results_dataset_csv2)

    ohit_df = pd.read_csv(ohit_dataset_csv)
    evaluate_ohit(ohit_df)

    # Read in the dataset.
    results_df = pd.read_csv(results_dataset_csv2)
    evaluate_results(results_df)
