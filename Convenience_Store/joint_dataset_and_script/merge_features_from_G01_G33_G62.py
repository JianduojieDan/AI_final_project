import pandas as pd
import os

print("Starting data merging process (G01, G33, G62 + OSM Features)...")

PROJECT_ROOT = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project"
BASE_PATH = os.path.join(PROJECT_ROOT, 'Convenience_Store', 'Data_for_Conven')

path_g01 = os.path.join(BASE_PATH, 'G01.conven.csv')  # population
path_g33 = os.path.join(BASE_PATH, 'G33.conven.csv')  # income
path_g62 = os.path.join(BASE_PATH, 'G62.conven.csv')  # transportation

path_osm_features = os.path.join(BASE_PATH, 'osm_features.csv')

output_path = os.path.join(BASE_PATH, 'MASTER_Convenience_Store_Dataset.csv')

try:
    df_g01 = pd.read_csv(path_g01)
    df_g33 = pd.read_csv(path_g33)
    df_g62 = pd.read_csv(path_g62)

    df_osm = pd.read_csv(path_osm_features)

    print(f"Loaded G01 (Population). Shape: {df_g01.shape}")
    print(f"Loaded G33 (Income).     Shape: {df_g33.shape}")
    print(f"Loaded G62 (Transport).  Shape: {df_g62.shape}")
    print(f"Loaded OSM Features.     Shape: {df_osm.shape}")

except FileNotFoundError as e:
    print(f"Error: A CSV file was not found. Details: {e}")
    print(f"Please check your paths in {BASE_PATH}")
    exit()

common_key = 'SA1_CODE_2021'

print(f"\nEnsuring all '{common_key}' keys are strings for merging...")
df_g01[common_key] = df_g01[common_key].astype(str)
df_g33[common_key] = df_g33[common_key].astype(str)
df_g62[common_key] = df_g62[common_key].astype(str)
df_osm[common_key] = df_osm[common_key].astype(str)

print("Starting merges...")

merged_df = pd.merge(df_g01, df_g33, on=common_key, how='inner')
print(f"After merging G01 + G33, shape is: {merged_df.shape}")

merged_df = pd.merge(merged_df, df_g62, on=common_key, how='inner')
print(f"After merging + G62, shape is: {merged_df.shape}")

final_df = pd.merge(merged_df, df_osm, on=common_key, how='inner')
print(f"After merging + OSM Features, final shape is: {final_df.shape}")

if final_df.empty:
    print("Error: The final dataframe is empty. No common 'SA1_CODE_2021' keys.")
else:
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccess! Master feature dataset saved to: {output_path}")
    print("\nNew Master Dataset Head (Top 5 rows):")
    print(final_df.head())