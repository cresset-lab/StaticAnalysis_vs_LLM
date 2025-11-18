import pandas as pd
import os
import requests
import ast
import re
from google import genai
from google.genai import types
import os
import time
from typing import Optional
from pathlib import Path
import os

PROJECT_ROOT = Path(r"C:\\Users\\jason\\OneDrive\Documents\\TMU\\CPS40A\\LLM-v-Static-Noura\\noura-openai")
RULES_BASE = PROJECT_ROOT
DATA_BASE = PROJECT_ROOT / "LLM-FP-verification"
CLOUD_KEY = ""

import re, ast
from pathlib import Path

def get_string_of_rules(base_path, file_paths, index, rule):
    try:
        rule = ast.literal_eval(rule)
    except Exception as e:
        print(f"[WARN] index {index}: bad rule format ({e}); skipping rules extraction.")
        return ""

    raw = Path(file_paths[index])                 # e.g., 'detect-output/iotb-rules/alarm.rules'
    full_path = raw if raw.is_absolute() else Path(base_path) / raw
    full_path = full_path.resolve()

    if not full_path.exists():
        print(f"[WARN] index {index}: rules file not found: {full_path} — continuing without it.")
        return ""

    string_of_rules, between_index = "", False
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if rule[0] in line or rule[1] in line:
                between_index = True
                string_of_rules += line.replace('\n', '\\n')
                continue
            if between_index:
                if re.search(r'\bend\b', line):
                    between_index = False
                string_of_rules += line.replace("\n", "\\n")
    return string_of_rules



def get_rit_prompt(df, index, prompt_folder, num_shot):
    # Get list of ohit rit detections
    ohit_results_list = list(df["trad_result"])

    # Get ohit detection for the current rule-pair
    ohit_result = ohit_results_list[index]

    # Define the prompt file based on the type of ohit result
    prompt_file = prompt_folder / f"{ohit_result.strip()}_FP_verify_prompt_{num_shot}.txt"

    # Open and read the prompt file
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt

from pathlib import Path
import re, unicodedata

PROJECT_ROOT = Path(r"C:\Users\jason\OneDrive\Documents\TMU\CPS40A\LLM-v-Static-Noura\noura-openai").resolve()
RULES_BASE   = PROJECT_ROOT
DATA_BASE    = PROJECT_ROOT / "LLM-FP-verification"

def _clean_cell_path(s: str) -> str:
    # normalize unicode, strip BOM, quotes, and whitespace
    s = s.replace('\ufeff', '')  # BOM if present
    s = unicodedata.normalize('NFKC', s)
    s = s.strip().strip('"').strip("'")
    # collapse backslashes to forward (Path handles both, this helps consistency)
    return s.replace('\\', '/')

def resolve_rules_path(cell_value: str) -> Path:
    raw_str = _clean_cell_path(cell_value)
    p = Path(raw_str)

    # If absolute, trust it
    if p.is_absolute():
        return p

    # If the CSV path starts with 'LLM-FP-verification/', drop that and anchor at PROJECT_ROOT
    parts = p.parts
    if len(parts) > 0 and parts[0].lower() == 'llm-fp-verification':
        p = PROJECT_ROOT.joinpath(*parts[1:])
    else:
        # typical case: 'detect-output/...'
        p = RULES_BASE / raw_str

    p = p.resolve()

    # If missing, try a targeted fallback search by filename under RULES_BASE
    if not p.exists():
        name = Path(raw_str).name  # e.g., 'alarm.rules'
        # search under detect-output only to keep it cheap
        candidates = list((RULES_BASE / "detect-output").rglob(name))
        if candidates:
            return candidates[0]

    return p


def get_string_of_rules(base_path, file_paths, index, rule):
    import ast, re
    try:
        rule = ast.literal_eval(rule)
    except Exception as e:
        print(f"[WARN] index {index}: bad rule format ({e}); skipping rules extraction.")
        return ""

    requested = str(file_paths[index])
    full_path = resolve_rules_path(requested)

    # Helpful diagnostics (repr shows hidden chars)
    if not full_path.exists():
        print(f"[WARN] index {index}: rules file not found:"
              f"\n  raw:    {repr(requested)}"
              f"\n  cleaned:{repr(_clean_cell_path(requested))}"
              f"\n  final:  {full_path}")
        return ""

    string_of_rules, between_index = "", False
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if rule[0] in line or rule[1] in line:
                between_index = True
                string_of_rules += line.replace('\n', '\\n')
                continue
            if between_index:
                if re.search(r'\bend\b', line):
                    between_index = False
                string_of_rules += line.replace("\n", "\\n")
    return string_of_rules


def get_results_file_path(rules_file_path: str) -> Path:
    """
    Given a path to a *.rules file, return the absolute Path to the corresponding
    results text file, e.g. detect-output/Results/<dataset>-results/<name>.txt.

    - Detects dataset type from the rules path ('iotb' => iotb-results, else oh-results)
    - Derives the base name from the *.rules filename
    - Returns an absolute Path
    - If not found, raises a detailed FileNotFoundError
    """
    rules_path = Path(rules_file_path)
    rules_lower = str(rules_path).lower()

    dataset_dir = "iotb-results" if "iotb" in rules_lower else "oh-results"

    # Preferred: use filename stem (foo.rules -> foo). This avoids regex brittleness.
    stem = rules_path.stem  # e.g., "alarm"

    # Build the absolute expected path
    results_dir = PROJECT_ROOT / "detect-output" / "Results" / dataset_dir
    candidate = results_dir / f"{stem}.txt"

    if candidate.exists():
        return candidate

    # (Optional) Try the regex your code used before, in case your CSV has odd paths
    # like ".../rules/Alarm.rules" and you want specifically the segment under /rules/
    m = re.search(r'(?<=rules[\\/]).*?(?=\.rules)', str(rules_path), flags=re.IGNORECASE)
    if m:
        alt = results_dir / f"{m.group(0)}.txt"
        if alt.exists():
            return alt

    # As a last resort, show the directory contents to help debug mismatches
    listed = []
    if results_dir.exists():
        listed = [p.name for p in results_dir.glob("*.txt")]

    raise FileNotFoundError(
        f"Results file not found.\n"
        f" Looked for: {candidate}\n"
        + (f" Also tried: {alt}\n" if m else "")
        + f" rules_file_path was: {rules_file_path}\n"
        f" Results directory exists: {results_dir.exists()} -> {results_dir}\n"
        f" Available result files here: {listed[:50]}"
    )



def get_actions_for_threat_pair(rules_file_path, threat_pair):
    results_file_path = get_results_file_path(rules_file_path)
    
    with open(results_file_path, 'r') as f:
        content = f.read()
    
    # Split the file into entries using a line of dashes as the separator.
    entries = re.split(r'-{5,}', content)
    entries = [entry.strip() for entry in entries if entry.strip()]
    
    for entry in entries:
        # Find the threat pair in the entry.
        pair_match = re.search(r'THREAT PAIR:\s*\((.*?)\)', entry)
        if pair_match:
            current_pair = pair_match.group(1).strip()
            if current_pair == threat_pair:
                # Extract the block between "CONTRADICTORY ACTIONS:" and "THREAT DESCRIPTION:"
                actions_block_match = re.search(
                    r'CONTRADICTORY ACTIONS:(.*?)(?=THREAT DESCRIPTION:)',
                    entry,
                    re.DOTALL
                )
                if actions_block_match:
                    actions_block = actions_block_match.group(1).strip()
                    # Remove all instances of action IDs (e.g. [r2a2])
                    cleaned_actions_block = re.sub(r'\[[^\]]+\]', '', actions_block)
                    cleaned_actions_block = cleaned_actions_block.strip()
                    return cleaned_actions_block
                else:
                    return "Actions block not found."
    
    return None



def get_triggers_for_threat_pair(rules_file_path, threat_pair):
    results_file_path = get_results_file_path(rules_file_path)
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the file into entries using a line of dashes as the separator.
    entries = re.split(r'-{5,}', content)
    entries = [entry.strip() for entry in entries if entry.strip()]

    for entry in entries:
        # Find the threat pair in the entry.
        pair_match = re.search(r'THREAT PAIR:\s*\((.*?)\)', entry)
        if pair_match:
            current_pair = pair_match.group(1).strip()  # e.g. "r1a1, r2a7"
            if current_pair == threat_pair:
                # Extract the block between "OVERLAPPING TRIGGERS:" and either
                # "OVERLAPPING CONDITIONS:" or "CONTRADICTORY ACTIONS:"
                triggers_block_match = re.search(
                    r'OVERLAPPING TRIGGERS:(.*?)(?=OVERLAPPING CONDITIONS:|CONTRADICTORY ACTIONS:)',
                    entry,
                    re.DOTALL
                )
                if triggers_block_match:
                    triggers_block = triggers_block_match.group(1).strip()
                    # Remove all instances of trigger IDs (e.g. [r1t2])
                    cleaned_triggers_block = re.sub(r'\[[^\]]+\]', '', triggers_block)
                    cleaned_triggers_block = cleaned_triggers_block.strip()
                    return cleaned_triggers_block
                else:
                    return "Triggers block not found."
    
    return None


def get_trigger_action_for_threat_pair(rules_file_path, threat_pair):
    results_file_path = get_results_file_path(rules_file_path)
    
    with open(results_file_path, 'r') as f:
        content = f.read()
    
    # Split the file into entries using a line of dashes as the separator.
    entries = re.split(r'-{5,}', content)
    entries = [entry.strip() for entry in entries if entry.strip()]
    
    for entry in entries:
        # Find the threat pair in the entry.
        pair_match = re.search(r'THREAT PAIR:\s*\((.*?)\)', entry)
        if pair_match:
            current_pair = pair_match.group(1).strip()
            if current_pair == threat_pair:
                # Extract the block between "TRIGGER-ACTION PAIR:" and either "RESULTING ACTION:" or "OVERLAPPING CONDITIONS:"
                block_match = re.search(
                    r'TRIGGER-ACTION PAIR:(.*?)(?=RESULTING ACTION:|OVERLAPPING CONDITIONS:)',
                    entry,
                    re.DOTALL
                )
                if block_match:
                    block = block_match.group(1).strip()
                    # Remove all instances of bracketed trigger/action IDs (e.g., [r1a1], [r2t1])
                    cleaned_block = re.sub(r'\[[^\]]+\]', '', block)
                    cleaned_block = cleaned_block.strip()
                    return cleaned_block
                else:
                    return "Trigger-action block not found."
    
    return None


def run_SAC_experiment(prompt_folder, current_file_path, current_threat_pair, num_shot):
    # Get the trigger information for the RIT
    threat_pair_triggers = get_triggers_for_threat_pair(current_file_path, current_threat_pair)
    
    # Retrieve the overlapping_triggers_prompt
    prompt_file = prompt_folder / f'overlapping_triggers_prompt_{num_shot}.txt'
    with open(prompt_file, 'r') as f:
        prompt_preface = f.read()

    # Combine the prompt with the trigger information
    overlapping_triggers_prompt = prompt_preface + str(threat_pair_triggers)
    
    # Send to LLM and get result
    overlapping_triggers_result = run_llm(overlapping_triggers_prompt)
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if "TP" in overlapping_triggers_result:
        # Get the action information for the RIT
        threat_pair_actions = get_actions_for_threat_pair(current_file_path, current_threat_pair)
        
        # Retrieve the conflicting_actions_prompt
        prompt_file = prompt_folder / f'conflicting_actions_prompt_{num_shot}.txt'
        with open(prompt_file, 'r') as f:
            prompt_preface = f.read()

        # Combine the prompt with the trigger information
        conflicting_actions_prompt = prompt_preface + str(threat_pair_actions)

        # Send to LLM and get result
        overlapping_actions_result = run_llm(conflicting_actions_prompt)

        # If both overlapping triggers and conflicting actions found:
        if "TP" in overlapping_actions_result:
            return "TP"

    return "FP"

def run_WAC_experiment(prompt_folder, current_file_path, current_threat_pair, num_shot):
    # Get the trigger information for the RIT
    threat_pair_triggers = get_triggers_for_threat_pair(current_file_path, current_threat_pair)
    
    # Retrieve the overlapping_triggers_prompt
    prompt_file = prompt_folder / f'overlapping_triggers_prompt_{num_shot}.txt'
    with open(prompt_file, 'r') as f:
        prompt_preface = f.read()

    # Combine the prompt with the trigger information
    overlapping_triggers_prompt = prompt_preface + str(threat_pair_triggers)
    
    # Send to LLM and get result
    overlapping_triggers_result = run_llm(overlapping_triggers_prompt)
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if "TP" in overlapping_triggers_result:
        # Get the action information for the RIT
        threat_pair_actions = get_actions_for_threat_pair(current_file_path, current_threat_pair)
        
        # Retrieve the conflicting_actions_prompt
        prompt_file = prompt_folder / f'conflicting_actions_prompt_{num_shot}.txt'
        with open(prompt_file, 'r') as f:
            prompt_preface = f.read()

        # Combine the prompt with the trigger information
        conflicting_actions_prompt = prompt_preface + str(threat_pair_actions)

        # Send to LLM and get result
        overlapping_actions_result = run_llm(conflicting_actions_prompt)

        # If both overlapping triggers and conflicting actions found:
        if "TP" in overlapping_actions_result:
            return "TP"

    return "FP"

def run_TC_experiment(prompt_folder, current_file_path, current_threat_pair, string_of_rules, num_shot):
    # Get the trigger information for the RIT
    threat_pair_trigger_action = get_trigger_action_for_threat_pair(current_file_path, current_threat_pair)
    
    # Retrieve the overlapping_triggers_prompt
    prompt_file = prompt_folder / f'trigger_cascade_prompt_{num_shot}.txt'
    with open(prompt_file, 'r') as f:
        prompt_preface = f.read()

    # Combine the prompt with the trigger information
    trigger_cascade_prompt = prompt_preface + "TRIGGER-ACTION PAIR:\n" + str(threat_pair_trigger_action) + "\n\nCANDIDATE RULES:\n" + string_of_rules
    
    # Send to LLM and get result
    trigger_cascade_result = run_llm(trigger_cascade_prompt)
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if "TP" in trigger_cascade_result:
        return "TP"

    return "FP"

def run_CC_experiment(prompt_folder, current_file_path, current_threat_pair, string_of_rules, num_shot):    
    # Retrieve the overlapping_triggers_prompt
    prompt_file = prompt_folder / f'condition_cascade_prompt_{num_shot}.txt'
    with open(prompt_file, 'r') as f:
        prompt_preface = f.read()

    # Combine the prompt with the trigger information
    condition_cascade_prompt = prompt_preface + string_of_rules
    
    # Send to LLM and get result
    condition_cascade_result = run_llm(condition_cascade_prompt)
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if "TP" in condition_cascade_result:
        return "TP"

    return "FP"


# One global client (connection reuse)
_GEMINI_CLIENT = genai.Client(api_key=CLOUD_KEY)

def _get_gemini_client() -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        # If you prefer explicit: genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        _GEMINI_CLIENT = genai.Client()
    return _GEMINI_CLIENT

def run_llm(prompt: str = '') -> str:
    """
    Sends a single text prompt to Gemini 2.5 Pro and returns the model's text.
    Keeps the same contract as your old function.
    """
    client = _get_gemini_client()

    # Tweak these to taste
    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
        top_p=0.95,
    )

    # Simple, non-streaming call
    # Model name per docs: "gemini-2.5-pro"
    # You could also try "gemini-2.5-flash" for speed/cost trades.
    last_err = None
    for attempt in range(5):  # light retry with exponential backoff
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=generation_config,
            )
            # SDK provides .text with concatenated parts
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt * 0.5)
    # If all retries fail, bubble up something sensible
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")

def run_experiment(path, dataframe, prompt_folder, num_shot):
    rules_list         = list(dataframe["rules"])
    file_paths_list    = list(dataframe["file_name"])
    ohit_results_list  = list(dataframe["trad_result"])
    threat_pairs_list  = list(dataframe["threat_pair"])

    llm_result_list = []
    for index, rule in enumerate(rules_list):
        try:
            print(f"Processing file index: {index}", flush=True)

            current_RIT_type   = ohit_results_list[index].strip()
            current_file_path  = file_paths_list[index]
            current_threat_pair= threat_pairs_list[index]

            if current_RIT_type == 'SAC':
                result = run_SAC_experiment(prompt_folder, current_file_path, current_threat_pair, num_shot)

            elif current_RIT_type == 'WAC':
                result = run_SAC_experiment(prompt_folder, current_file_path, current_threat_pair, num_shot)

            elif current_RIT_type[1:] == 'TC':
                string_of_rules = get_string_of_rules(path, file_paths_list, index, rule)
                if not string_of_rules:
                    print(f"[WARN] index {index}: no rules text; marking FP and continuing.")
                    result = "FP"
                else:
                    result = run_TC_experiment(prompt_folder, current_file_path, current_threat_pair, string_of_rules, num_shot)

            elif current_RIT_type[1:] == 'CC':
                string_of_rules = get_string_of_rules(path, file_paths_list, index, rule)
                if not string_of_rules:
                    print(f"[WARN] index {index}: no rules text; marking FP and continuing.")
                    result = "FP"
                else:
                    result = run_CC_experiment(prompt_folder, current_file_path, current_threat_pair, string_of_rules, num_shot)

            else:
                print(f"[WARN] index {index}: unknown RIT type '{current_RIT_type}'; marking FP.")
                result = "FP"

        except Exception as e:
            # Never crash the run; record a conservative default
            import traceback
            print(f"[ERROR] index {index}: {e}\n{traceback.format_exc().rstrip()}")
            result = "FP"

        llm_result_list.append(result)

        # Optional: checkpoint partial progress every row to survive restarts
        dataframe.loc[index, 'truth_experiment'] = result
        temp_csv = f"{output_csv_prefix}_{num_shot}_partial.csv"
        dataframe.to_csv(temp_csv, index=False)

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
    You might need to edit:
        the prompt_folder, dataset_csv path, output_csv_prefix for your system
        run_llm() function to use llama
    '''

    

    # May need to change these
    prompt_folder = DATA_BASE  / "prompts_fp" / "v2"
    dataset_csv = DATA_BASE  / "datasets_fp" / "dataset_ohit_fp.csv"
    output_csv_prefix = DATA_BASE  / "results" / "NEW_gem_25_results"

    # To iterate through and select zero, one, or two-shot prompt
    # experiment_type = ['0', '1', '2']
    experiment_type = ['1', '2']

    path = RULES_BASE

    # Iterate through each file in the prompt folder.
    for num_shot in experiment_type:
        # Read in the dataset.
        df = pd.read_csv(str(dataset_csv))

        # Process the prompt with your custom function.
        llm_result_list = run_experiment(path, df, prompt_folder, num_shot)

        # Add the result list to the DataFrame.
        df['truth_experiment'] = llm_result_list

        evaluate_results(df)

        # Generate an output CSV file name that includes the prompt file name.
        output_csv = f"{output_csv_prefix}_{num_shot}.csv"
        df.to_csv(str(output_csv), index=False)

        print(f"Created CSV: {output_csv}")

