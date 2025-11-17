import pandas as pd
import requests
import json
import os
"""
Perform LLM classification using a local LLM API and save the results to CSV files.
"""

def apply_api(path='', dataframe='', prompt=''):
    url = "http://localhost:46495/api/generate"
    ruleset = list(dataframe['ruleset'])
    llm_result = []
    for rule in ruleset:
    # api call
        p = {
            "model": "llama3.1:8b",
            "prompt": prompt + rule + "\nCan we classify these openHAB rulesets as any of the rule interaction threats mentioned before?",
            "options": {
                "num_ctx": 4096
            }
        } 
        #response = requests.post(url, json=p)
        with requests.post(url, json=p, stream=True) as response: 
            full_text= ""
            #print(response)
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try: 
                        data = json.loads(line)
                        token = data.get("response", "")
                        full_text += token
                        #print(token, end="", flush=True)
                       # llm_result.append(token) 
                    except json.JSONDecodeError:
                        print("\nCan't parse line:", line)
        llm_result.append(full_text) # add the response the the llm rule list
        print(full_text)
        print("\nDone")
    return llm_result
if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    prompt = ''
    i = 0
    base_dir = ''
    for root, _, files in os.walk(base_dir): # we iterate through all the folders of the base directory. these directories are experiment a, experiment b, experiment c etc.
        if 'experiment' in root and 'experiment_e' not in root and 'results' not in root: # we don't want the results folder to be included in this or experiment e (since the prompts have to be different)
        #if 'experiment_d' in root or 'experiment_c' in root nd 'results' not in root:
            for file in files: 
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    prompt = f.read()
                    llm_result_list = apply_api(dataframe=df, prompt=prompt) # apply the api to the
                    df['LLM_result'] = llm_result_list
                    df.to_csv(root  + '/results/8b_model/' + file  +'VERSION3_RESULTS.csv', index=None) # change this every time i run a new version of the overrall experiment
                    print('saved', str(i), file_path)
                    i += 1
    '''
    i = 1 
    files = ['/home/nkhajehn/summer_research_v2/experiment_d/prompt_2letter_1shot_multiple.txt', '/home/nkhajehn/summer_research_v2/experiment_d/prompt_2letter_2shot_multiple.txt']  
    for file_path in files:
        with open(file_path) as f:
            prompt = f.read()
            llm_result_list = apply_api(dataframe=df, prompt=prompt)
            df['LLM_result'] = llm_result_list 
            df.to_csv('/home/nkhajehn/summer_research_v2/experiment_d' + '/results/' + str(i) + '_RESULTS.csv', index=None)
            i += 1
                    
    
    with open('/home/nkhajehn/summer_research_v2/experiment_a/prompt_3letter_1shot_nodups.txt') as f:
        prompt = f.read()
    llm_result_list = apply_api(path='/home/nkhajehn/', dataframe=df, prompt=prompt)

    df['LLM_result'] = llm_result_list
    df.to_csv('llama_70b_experimenta_1shot_nodups_v1.csv', index=None)
    '''
