import pandas as pd

def label_ohit_file(groundtruth_file, ohit_file, output_file):
    # Read the CSV files
    df_gt = pd.read_csv(groundtruth_file)
    df_oh = pd.read_csv(ohit_file)
    
    # Merge the ohit dataframe with the ground truth on all columns.
    # Rows that match in both will have _merge == 'both', others 'left_only'
    df_merged = df_oh.merge(df_gt.drop_duplicates(), on=list(df_gt.columns), how='left', indicator=True)
    
    # Label as 'TP' if the row is found in the groundtruth (i.e., _merge == 'both'), else 'FP'
    df_merged['truth_actual'] = df_merged['_merge'].apply(lambda x: 'TP' if x == 'both' else 'FP')
    
    # Optionally drop the _merge column used for the merge indicator
    df_merged.drop(columns=['_merge'], inplace=True)
    
    # Write the result to a new CSV file
    df_merged.to_csv(output_file, index=False)
    print(f"Labeled data saved to {output_file}")

# Hardcoded file paths for when you're running the script in VSCode:
groundtruth_file = "dataset_groundtruth.csv"
ohit_file = "dataset_ohit.csv"
output_file = "dataset_ohit_FP.csv"

label_ohit_file(groundtruth_file, ohit_file, output_file)
