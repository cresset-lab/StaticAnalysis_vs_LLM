import pandas as pd

def analyze_results(file_path):
    """
    Analyzes the results from a CSV file and displays the analysis.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Get the unique values in the 'trad_result' column
    trad_results = df['trad_result'].unique()

    # --- Analysis and Display ---
    print("--- Analysis of Truth Values by trad_result ---")

    summary_data = []

    for result in sorted(trad_results):
        print(f"\n--- trad_result: {result} ---")
        subset = df[df['trad_result'] == result]

        # --- Counts for Truth Value Combinations ---
        tp_tp = len(subset[(subset['truth_actual'] == 'TP') & (subset['truth_experiment'] == 'TP')])
        tp_fp = len(subset[(subset['truth_actual'] == 'TP') & (subset['truth_experiment'] == 'FP')])
        fp_tp = len(subset[(subset['truth_actual'] == 'FP') & (subset['truth_experiment'] == 'TP')])
        fp_fp = len(subset[(subset['truth_actual'] == 'FP') & (subset['truth_experiment'] == 'FP')])

        # --- Display in a Table-like Format ---
        print(f"{'':<15} | {'Experiment: TP':<15} | {'Experiment: FP':<15}")
        print("-" * 50)
        print(f"{'Actual: TP':<15} | {tp_tp:<15} | {tp_fp:<15}")
        print(f"{'Actual: FP':<15} | {fp_tp:<15} | {fp_fp:<15}")

        # --- Calculations for the new table ---
        correct = tp_tp + fp_fp
        incorrect = tp_fp + fp_tp
        
        # Precision
        if (tp_tp + fp_tp) > 0:
            precision = tp_tp / (tp_tp + fp_tp)
        else:
            precision = 0.0
            
        # Recall
        if (tp_tp + tp_fp) > 0:
            recall = tp_tp / (tp_tp + tp_fp)
        else:
            recall = 0.0
            
        summary_data.append({
            'trad_result': result,
            'correct': correct,
            'incorrect': incorrect,
            'precision': f"{precision:.2f}",
            'recall': f"{recall:.2f}"
        })

    # --- Display the new summary table ---
    print("\n\n--- Summary Table ---")
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


    # --- Mismatched Entries ---
    print("\n\n--- Mismatched Entries (truth_actual vs. truth_experiment) ---")
    mismatches = df[df['truth_actual'] != df['truth_experiment']]

    if mismatches.empty:
        print("No mismatched entries found.")
    else:
        for index, row in mismatches.iterrows():
            print("\n")
            print(f"File Name:          {row['file_name']}")
            print(f"Rules:              {row['rules']}")
            print(f"Threat Pair:        {row['threat_pair']}")
            print(f"Trad Result:        {row['trad_result']}")
            print(f"Truth (Actual):     {row['truth_actual']}")
            print(f"Truth (Experiment): {row['truth_experiment']}")
            print("-" * 40)


# --- Create a dummy CSV for demonstration ---
data = {
    'file_name': [f'file_{i}.txt' for i in range(1, 13)],
    'rules': ['rule_a', 'rule_b', 'rule_c'] * 4,
    'threat_pair': [f'pair_{i}' for i in range(1, 13)],
    'trad_result': ['WAC', 'SAC', 'STC', 'WTC', 'WCC', 'SCC'] * 2,
    'truth_actual': ['TP', 'TP', 'FP', 'FP', 'TP', 'FP'] * 2,
    'truth_experiment': ['TP', 'FP', 'TP', 'FP', 'TP', 'TP'] * 2,
}
dummy_df = pd.DataFrame(data)
dummy_df.to_csv("analysis_results.csv", index=False)


# --- Specify the CSV file and run the analysis ---
# csv_file_to_analyze = r"LLM-v-Static-Noura\\noura-openai\\LLM-FP-verification\\results\\gpt4o_fp_results_2.csv"
csv_file_to_analyze = r"LLM-v-Static-Noura\\noura-openai\\LLM-FP-verification\\results\\NEW_gem_25_results_2.csv"
analyze_results(csv_file_to_analyze)