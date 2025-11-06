import pandas as pd
import os

print("Starting data merging process (G01, G33, G62)...")

BASE_PATH = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/Convenience Store/Data_for_Conven/"

path_g01 = os.path.join(BASE_PATH, 'G01.conven.csv')  # population
path_g33 = os.path.join(BASE_PATH, 'G33.conven.csv')  # income
path_g62 = os.path.join(BASE_PATH, 'G62.conven.csv')  # transportation

output_path = os.path.join(BASE_PATH, 'MASTER_Convenience_Store_Dataset.csv')

try:
    df_g01 = pd.read_csv(path_g01)
    df_g33 = pd.read_csv(path_g33)
    df_g62 = pd.read_csv(path_g62)

    print(f"Loaded G01. Shape: {df_g01.shape}")
    print(f"Loaded G33. Shape: {df_g33.shape}")
    print(f"Loaded G62. Shape: {df_g62.shape}")

except FileNotFoundError:
    print(f"Error: One of the CSV files (G01, G33, G62) was not found in {BASE_PATH}")
    exit()

common_key = 'SA1_CODE_2021'

merged_df = pd.merge(df_g01, df_g33, on=common_key, how='inner')
print(f"After merging G01 + G33, shape is: {merged_df.shape}")

final_df = pd.merge(merged_df, df_g62, on=common_key, how='inner')
print(f"After merging (G01+G33) + G62, final shape is: {final_df.shape}")

if final_df.empty:
    print("Error: The final dataframe is empty. No common 'SA1_CODE_2021' keys.")
else:
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccess! Master feature dataset saved to: {output_path}")