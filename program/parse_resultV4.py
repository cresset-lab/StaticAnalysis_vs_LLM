import os
import pandas as pd
import traceback


def count_rit_types(text, rit_list):
    """Count how many different RIT types appear in the text"""
    count = 0
    for rit in rit_list:
        if rit in text:
            count += 1
    return count


def post_processing_results(df, file_name):
    correct, incorrect = 0, 0 
    wac_correct, sac_correct, wtc_correct, stc_correct, wcc_correct, scc_correct = 0, 0, 0, 0, 0, 0
    wac_incorrect, sac_incorrect, stc_incorrect, wtc_incorrect, wcc_incorrect, scc_incorrect = 0, 0, 0, 0, 0, 0
    ac_correct, cc_correct, tc_correct = 0, 0, 0
    ac_incorrect, cc_incorrect, tc_incorrect = 0, 0, 0
    explanation_pred = 0 
    predictions = list(df['LLM_result'])
    actual = list(df['trad_result'])
    # Determine which experiment type based on file path
    is_experiment_b = 'experiment_b' in file_name
    is_experiment_d = 'experiment_d' in file_name

    for index, prediction in enumerate(predictions):
        actual_val = str(actual[index]).strip()
        
        # Extract the answer portion after markers
        pred_val = str(prediction) #extract_answer(prediction)
        
        # Check for multiple RIT types in the prediction
        multiple_rits_detected = False
        
        if is_experiment_b:
            # For experiment_b: check if more than 1 of the 6 3-letter RITs appear
            three_letter_rits = ['WAC', 'SAC', 'WTC', 'STC', 'WCC', 'SCC']
            rit_count = count_rit_types(pred_val, three_letter_rits)
            if rit_count > 1:
                multiple_rits_detected = True
        
        elif is_experiment_d:
            # For experiment_d: check if more than 1 of the 3 2-letter RITs appear
            two_letter_rits = ['AC', 'TC', 'CC']
            rit_count = count_rit_types(pred_val, two_letter_rits)
            if rit_count > 1:
                multiple_rits_detected = True
        
        # If multiple RITs detected, mark as incorrect immediately
        if multiple_rits_detected:
            incorrect += 1
            
            # Increment 3-letter counter
            if actual_val == 'WAC':
                wac_incorrect += 1
            elif actual_val == 'SAC':
                sac_incorrect += 1
            elif actual_val == 'WTC':
                wtc_incorrect += 1
            elif actual_val == 'STC':
                stc_incorrect += 1
            elif actual_val == 'WCC':
                wcc_incorrect += 1
            elif actual_val == 'SCC':
                scc_incorrect += 1

            # Increment 2-letter counter
            if actual_val in ['WAC', 'SAC']:
                ac_incorrect += 1
            elif actual_val in ['WTC', 'STC']:
                tc_incorrect += 1
            elif actual_val in ['WCC', 'SCC']:
                cc_incorrect += 1
            
            # Check for long text response
            if any(len(word) > 3 for word in pred_val.split(',')):
                explanation_pred += 1
            
            continue  # Skip to next prediction

        if '3letter' in file_name:
            if actual_val in pred_val:
                correct += 1
                
                # Only increment the counter for the matching category
                if actual_val == 'WAC':
                    wac_correct += 1
                elif actual_val == 'SAC':
                    sac_correct += 1
                elif actual_val == 'WTC':
                    wtc_correct += 1
                elif actual_val == 'STC':
                    stc_correct += 1
                elif actual_val == 'WCC':
                    wcc_correct += 1
                elif actual_val == 'SCC':
                    scc_correct += 1
            
            else:
                incorrect += 1
                
                # Increment incorrect counter for the actual category
                if actual_val == 'WAC':
                    wac_incorrect += 1
                elif actual_val == 'SAC':
                    sac_incorrect += 1
                elif actual_val == 'WTC':
                    wtc_incorrect += 1
                elif actual_val == 'STC':
                    stc_incorrect += 1
                elif actual_val == 'WCC':
                    wcc_incorrect += 1
                elif actual_val == 'SCC':
                    scc_incorrect += 1
            if any(len(word) > 3 for word in pred_val.split(',')):
                    # If the responses contain workds with length > 3, then it is considered a long text response
                    explanation_pred += 1
            
        else:
            # 2-letter mode
            if actual_val in ['WAC', 'SAC']:
                if 'AC' in pred_val:
                    ac_correct += 1 
                    correct += 1 
                else: 
                    ac_incorrect += 1
                    incorrect += 1
                    
            elif actual_val in ['WTC', 'STC']:
                if 'TC' in pred_val:
                    tc_correct += 1 
                    correct += 1 
                else:
                    tc_incorrect += 1 
                    incorrect += 1
                    
            elif actual_val in ['WCC', 'SCC']:  
                if 'CC' in pred_val:
                    cc_correct += 1 
                    correct += 1 
                else: 
                    cc_incorrect += 1 
                    incorrect += 1
            if any(len(word) > 3 for word in pred_val.split(',')):
                    # If the responses contain workds with length > 3, then it is considered a long text response
                    explanation_pred += 1
        
    # Print results with better formatting
    print("=" * 80)
    print(f"File: {file_name}")
    print("=" * 80)
    
    if '3letter' in file_name:
        total = correct + incorrect
        print(f"Correct:    {correct:4d}  ({correct/total*100:.2f}%)")
        print(f"Incorrect:  {incorrect:4d}  ({incorrect/total*100:.2f}%)")
        print(f"Long text:  {explanation_pred:4d}  ({explanation_pred/total*100:.2f}%)")
        print(f"Total:      {total:4d}")
        print()
        
        # Only print category stats if there are samples for that category
        if wac_correct + wac_incorrect > 0:
            print(f"WAC accuracy: {wac_correct/(wac_correct + wac_incorrect)*100:.2f}% ({wac_correct}/{wac_correct + wac_incorrect})")
        if sac_correct + sac_incorrect > 0:
            print(f"SAC accuracy: {sac_correct/(sac_correct + sac_incorrect)*100:.2f}% ({sac_correct}/{sac_correct + sac_incorrect})")
        if wtc_correct + wtc_incorrect > 0:
            print(f"WTC accuracy: {wtc_correct/(wtc_correct + wtc_incorrect)*100:.2f}% ({wtc_correct}/{wtc_correct + wtc_incorrect})")
        if stc_correct + stc_incorrect > 0:
            print(f"STC accuracy: {stc_correct/(stc_correct + stc_incorrect)*100:.2f}% ({stc_correct}/{stc_correct + stc_incorrect})")
        if wcc_correct + wcc_incorrect > 0:
            print(f"WCC accuracy: {wcc_correct/(wcc_correct + wcc_incorrect)*100:.2f}% ({wcc_correct}/{wcc_correct + wcc_incorrect})")
        if scc_correct + scc_incorrect > 0:
            print(f"SCC accuracy: {scc_correct/(scc_correct + scc_incorrect)*100:.2f}% ({scc_correct}/{scc_correct + scc_incorrect})")
    else: 
        total = correct + incorrect
        print(f"Correct:    {correct:4d}  ({correct/total*100:.2f}%)")
        print(f"Incorrect:  {incorrect:4d}  ({incorrect/total*100:.2f}%)")
        print(f"Long text:  {explanation_pred:4d}  ({explanation_pred/(total + explanation_pred)*100:.2f}%)")
        print(f"Total:      {total:4d}")
        print()
        
        if ac_correct + ac_incorrect > 0:
            print(f"AC accuracy: {ac_correct/(ac_correct + ac_incorrect)*100:.2f}% ({ac_correct}/{ac_correct + ac_incorrect})")
        if tc_correct + tc_incorrect > 0:
            print(f"TC accuracy: {tc_correct/(tc_correct + tc_incorrect)*100:.2f}% ({tc_correct}/{tc_correct + tc_incorrect})")
        if cc_correct + cc_incorrect > 0:
            print(f"CC accuracy: {cc_correct/(cc_correct + cc_incorrect)*100:.2f}% ({cc_correct}/{cc_correct + cc_incorrect})")
    
    print()


if __name__ == '__main__':
    base_dir = r''
    bad_files = []
    processed_files = []

    for root, _, files in os.walk(base_dir):
        # if 'results' in root and 'experiment_d' in root and 'gemini' in root and 'mutation_dataset' in root:
        if 'results' in root and 'experiment_c' in root and 'gemini' in root:
            for file in files:
                file_path = os.path.join(root, file)

                if not file_path.lower().endswith('.csv'):
                    continue
                if 'mutation' in file_path.lower(): # Control the specific files to process
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError as e:
                        print(f"[ENCODING ERROR] {file_path}")
                        print(f"  -> {e}")
                        bad_files.append(file_path)
                        continue
                    except Exception as e:
                        print(f"[READ ERROR] {file_path}")
                        print(f"  -> {type(e).__name__}: {e}")
                        bad_files.append(file_path)
                        continue

                    try:
                        post_processing_results(df, file_path)
                        processed_files.append(file_path)
                    except Exception as e:
                        print(f"[PROCESS ERROR] {file_path}")
                        print(f"  -> {type(e).__name__}: {e}")
                        traceback.print_exc()
                        continue

    print("\n" + "=" * 80)
    print(f"SUMMARY: Processed {len(processed_files)} files successfully")
    
    if bad_files:
        print(f"\nFailed to read {len(bad_files)} files:")
        for p in bad_files:
            print(f"  - {p}")