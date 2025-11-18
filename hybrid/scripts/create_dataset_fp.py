

import pandas as pd
import os
import requests
import ast
import transformers
import torch 
import re


def extract_info_from_benchmark_results(path, dict_of_results):
    for folder in os.listdir(path):
        inner_folder_path = path + '/' + folder
        if os.path.isdir(inner_folder_path):
            for filename in os.listdir(inner_folder_path):
                file_path = inner_folder_path + "/" + filename
                if filename[-3:] == 'txt':
                    with open(file_path, 'r') as file:
                        rule_values = ['a', 'b']
                        threat_pair = ''
                        name = ""
                        for index, line in enumerate(file):
                            if index == 0: 
                                name = line[6:-1]
                            if "RULE_A" in line: 
                                rule_values[0] = line[line.index('(') + 1:-2]
                            elif "RULE_B" in line: 
                                rule_values[1] = line[line.index('(') + 1: -2]
                            elif "THREAT PAIR" in line:
                                threat_pair = line[line.index('(') + 1: -2]
                            elif rule_values[0] != 'a' and rule_values[1] != 'b' and '------' in line:
                                dict_of_results["rules"].append(rule_values)
                                dict_of_results["file_name"].append(name.replace('\\', '/'))
                                dict_of_results["threat_pair"].append(threat_pair)
                                rule_values = ['a', 'b']
                            elif "THREAT DETECTED" in line:
                                index_of_result = line.index(". ")
                                dict_of_results["trad_result"].append(line[index_of_result + 1: index_of_result + 5])

            
if __name__ == "__main__":

    # for parsing detect output and making it into csv file 
    
    dict_of_results = {"file_name": [], "rules": [], "threat_pair": [], "trad_result": []}

    path = 'detect-output/Results-GroundTruth'
    extract_info_from_benchmark_results(path, dict_of_results)

    df = pd.DataFrame(dict_of_results)

    df.to_csv('dataset_groundtruth.csv', index=None)
