import pandas as pd
import os
import requests
import ast
import re
from google import genai
from google.genai import types
import os
import time
import json
from typing import Optional
from pathlib import Path
import os

PROJECT_ROOT = path_prefix = Path(r"C:\\Users\jason\\OneDrive\Documents\\TMU\CPS40A\\LLM-v-Static-Noura\\noura-openai")
CLOUD_KEY = ""

LOG_DIR = PROJECT_ROOT / "LLM-FP-verification" / "llm_logs.txt"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_GEMINI = None

def _get_client():
    global _GEMINI
    if _GEMINI is None:
        _GEMINI = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _GEMINI

def _append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run_llm_structured(prompt: str,
                       model: str = "gemini-2.5-pro",
                       run_tag: str = "default",
                       extra_meta: dict | None = None):
    """
    Ask for a concise, structured explanation (NOT chain-of-thought).
    Returns (verdict_text, full_record_dict).
    Also appends the full record to llm_logs/<run_tag>.jsonl
    """
    client = _get_client()

    # We tell the model to return a short, structured rationale (not step-by-step thoughts)
    system_preamble = (
        "You are a verification assistant. Respond in strict JSON with fields:\n"
        "{"
        "  'verdict': 'TP' or 'FP',"
        "  'reason_brief': '1-3 sentences explaining the decision at a high level',"
        "  'evidence': ['short quotes or normalized facts you used'],"
        "  'confidence': 0.0-1.0"
        "}\n"
        "Do not include hidden reasoning. Be concise."
    )

    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,
        top_p=0.9,
        response_mime_type="application/json",  # JSON-mode
    )

    t0 = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=[system_preamble, "\n\nUSER_PROMPT:\n", prompt],
        config=cfg,
    )
    latency_s = time.time() - t0

    text = (resp.text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback if model misbehaves: treat body as plain verdict probe
        data = {
            "verdict": "TP" if "TP" in text else ("FP" if "FP" in text else "UNKNOWN"),
            "reason_brief": "Non-JSON response; captured raw text.",
            "evidence": [text[:400]],
            "confidence": 0.0
        }

    # Compose a record to persist
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "latency_s": round(latency_s, 3),
        "prompt_len": len(prompt),
        "response": data,
    }
    if extra_meta:
        record["meta"] = extra_meta

    # Write one line per call
    log_file = LOG_DIR / f"{run_tag}.jsonl"
    _append_jsonl(log_file, record)

    return (str(data.get("verdict", "")).strip(), record)


def _get_gemini_client():
    return genai.Client(
        vertexai=True,
        project="YOUR_GCP_PROJECT_ID",   # from the top bar in Cloud Console
        location="us-central1"           # or your chosen region
    )

def get_rit_prompt(df, index, prompt_folder, num_shot):
    # Get list of ohit rit detections
    ohit_results_list = list(df["trad_result"])

    # Get ohit detection for the current rule-pair
    ohit_result = ohit_results_list[index]

    # Define the prompt file based on the type of ohit result
    prompt_file = prompt_folder / ohit_result.lstrip() / f'_FP_verify_prompt_{num_shot}.txt'

    # Open and read the prompt file
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt

def get_string_of_rules(path, file_paths, index, rule):
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
                    #print(line)
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
    verdict, rec = run_llm_structured(
        overlapping_triggers_prompt,
        run_tag=f"SAC_overlapping_triggers_{num_shot}",
        extra_meta={"file": current_file_path, "pair": current_threat_pair}
    )
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if verdict == "TP":
        # Get the action information for the RIT
        threat_pair_actions = get_actions_for_threat_pair(current_file_path, current_threat_pair)
        
        # Retrieve the conflicting_actions_prompt
        prompt_file = prompt_folder / f'conflicting_actions_prompt_{num_shot}.txt'
        with open(prompt_file, 'r') as f:
            prompt_preface = f.read()

        # Combine the prompt with the trigger information
        conflicting_actions_prompt = prompt_preface + str(threat_pair_actions)

        # Send to LLM and get result
        verdict2, rec2 = run_llm_structured(
            conflicting_actions_prompt,
            run_tag=f"SAC_conflicting_actions_{num_shot}",
            extra_meta={"file": current_file_path, "pair": current_threat_pair}
        )

        # If both overlapping triggers and conflicting actions found:
        if verdict2 == "TP":
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
    verdict, rec = run_llm_structured(
        overlapping_triggers_prompt,
        run_tag=f"WAC_overlapping_triggers_{num_shot}",
        extra_meta={"file": current_file_path, "pair": current_threat_pair}
    )
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if verdict == "TP":
        # Get the action information for the RIT
        threat_pair_actions = get_actions_for_threat_pair(current_file_path, current_threat_pair)
        
        # Retrieve the conflicting_actions_prompt
        prompt_file = prompt_folder / f'conflicting_actions_prompt_{num_shot}.txt'
        with open(prompt_file, 'r') as f:
            prompt_preface = f.read()

        # Combine the prompt with the trigger information
        conflicting_actions_prompt = prompt_preface + str(threat_pair_actions)

        # Send to LLM and get result
        verdict2, rec2 = run_llm_structured(
            conflicting_actions_prompt,
            run_tag=f"WAC_conflicting_actions_{num_shot}",
            extra_meta={"file": current_file_path, "pair": current_threat_pair}
        )

        # If both overlapping triggers and conflicting actions found:
        if verdict2 == "TP":
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
    verdict, rec = run_llm_structured(
        trigger_cascade_prompt,
        run_tag=f"trigger_cascade_{num_shot}",
        extra_meta={"file": current_file_path, "pair": current_threat_pair}
    )
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if verdict == "TP":
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
    verdict, rec2 = run_llm_structured(
        condition_cascade_prompt,
        run_tag=f"condition_cascade_prompt_{num_shot}",
        extra_meta={"file": current_file_path, "pair": current_threat_pair}
    )
    
    # If overlapping triggers were identified, continue on to check for conflicting actions
    if verdict == "TP":
        return "TP"

    return "FP"


# One global client (connection reuse)
_GEMINI_CLIENT: Optional[genai.Client] = None

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
    # Get lists for indexing
    rules_list = list(dataframe["rules"])
    file_paths_list = list(dataframe["file_name"])
    ohit_results_list = list(df["trad_result"])
    threat_pairs_list = list(df["threat_pair"])

    llm_result_list = [] 
    stop_after = 0
    for index, rule in enumerate(rules_list):
        print(f"Processing file index: {index}")

        current_RIT_type = ohit_results_list[index].strip()
        current_file_path = file_paths_list[index]
        current_threat_pair = threat_pairs_list[index]

        
        if current_RIT_type == 'SAC':
            SAC_result = run_SAC_experiment(prompt_folder, current_file_path, current_threat_pair, num_shot)
            llm_result_list.append(SAC_result)
            
        elif current_RIT_type == 'WAC':
            WAC_result = run_SAC_experiment(prompt_folder, current_file_path, current_threat_pair, num_shot)
            llm_result_list.append(WAC_result)
            
        elif current_RIT_type[1:] == 'TC':
            # Get string of rules (rule_A, rule_B)
            string_of_rules = get_string_of_rules(path, file_paths_list, index, rule)

            TC_result = run_TC_experiment(prompt_folder, current_file_path, current_threat_pair, string_of_rules, num_shot)
            llm_result_list.append(TC_result)

        elif current_RIT_type[1:] == 'CC':
            # Get string of rules (rule_A, rule_B)
            string_of_rules = get_string_of_rules(path, file_paths_list, index, rule)

            CC_result = run_CC_experiment(prompt_folder, current_file_path, current_threat_pair, string_of_rules, num_shot)
            llm_result_list.append(CC_result)
        stop_after+=1
        if stop_after > 2: 
            with open(PROJECT_ROOT / "LLM-FP-verification" / "TEST.TXT", "w") as f:
                for item in llm_result_list:
                    f.write(item + "\n")
            break

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
    prompt_folder = PROJECT_ROOT / "LLM-FP-verification" / "prompts_fp" / "v2"
    dataset_csv = PROJECT_ROOT / "LLM-FP-verification" / "datasets_fp" / "dataset_ohit_fp.csv"
    output_csv_prefix = PROJECT_ROOT / "LLM-FP-verification" / "results" / "NEW_gem_25_results"

    # To iterate through and select zero, one, or two-shot prompt
    experiment_type = ['0', '1', '2']

    path = ''

    # Iterate through each file in the prompt folder.
    for num_shot in experiment_type:
        # Read in the dataset.
        df = pd.read_csv(dataset_csv)

        # Process the prompt with your custom function.
        llm_result_list = run_experiment(path, df, prompt_folder, num_shot)

        # Add the result list to the DataFrame.
        df['truth_experiment'] = llm_result_list

        evaluate_results(df)

        # Generate an output CSV file name that includes the prompt file name.
        output_csv = f'{output_csv_prefix}_{num_shot}.csv'
        df.to_csv(output_csv, index=False)

        print(f"Created CSV: {output_csv}")

