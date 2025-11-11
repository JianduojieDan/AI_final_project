import pandas as pd
import geopandas as gpd
import json
import os


def extract_g01_data(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Config file '{config_path}' is not formatted correctly.")
        return

    print("Config file read successfully. Looking for 'G62' settings...")
    try:
        settings = config['G62']
        file_key = 'G62'
    except KeyError:
        print("Error: Could not find 'G62' key in your config.json file.")
        return

    input_file = settings['input_path']
    output_file = settings['output_path']
    columns = settings['columns_to_extract']
    layer_name = settings.get('layer_name')

    print(f"\n--- Processing: {file_key} ---")

    try:
        print(f"Reading GeoPackage: {input_file} (Layer: {layer_name}, Engine: fiona)")
        df = gpd.read_file(input_file, layer=layer_name, engine="fiona")

        print("\n--- DEBUG: File read successfully. ---")
        print(f"Total columns found: {len(df.columns)}")
        print("Listing all column names found in the file:")
        print(list(df.columns))
        print("------------------------------------------")

        print("\nChecking for requested columns...")
        missing_cols = [col for col in columns if col not in df.columns]

        if missing_cols:
            print(f"Warning: The following columns were not found in {input_file}: {missing_cols}")
            columns_to_keep = [col for col in columns if col in df.columns]
        else:
            print("All requested columns were found.")
            columns_to_keep = columns

        if not columns_to_keep:
            print(f"Error: None of the requested columns were found in {input_file}. Skipping.")
            return

        print(f"Extracting {len(columns_to_keep)} columns...")
        df_selected = df[columns_to_keep]

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_selected.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Success! Extracted data for {file_key} saved to: {output_file}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print(f"Error Type: {type(e)}")
        print("--- G62 processing incomplete ---")
    else:
        print("\n--- G62 processing complete ---")


if __name__ == "__main__":
    config_file_path = "config.json"
    extract_g01_data(config_file_path)