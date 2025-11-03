import geopandas as gpd
import os
import pandas as pd

file_path_g62 = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/data/Geopackage_2021_G62_NSW_GDA2020/G62_NSW_GDA2020.gpkg"
layer_name_g62 = "G62_SA1_2021_NSW"

print(f"Attempting to read file: {file_path_g62}, Layer: {layer_name_g62}")

if os.path.exists(file_path_g62):
    try:
        gdf_g62_raw = gpd.read_file(file_path_g62, layer=layer_name_g62)
        print("\n--- Raw file reading successful! --- ✅")
        print(f"Raw data shape: {gdf_g62_raw.shape}")

        print("\nSelecting and renaming key columns...")
        columns_to_keep = {
            'SA1_CODE_2021': 'sa1_code',
            'Tot_P': 'total_employed_persons', # Total persons commuting/working
            'One_method_Train_P': 'commute_train',
            'One_method_Bus_P': 'commute_bus',
            'One_method_Ferry_P': 'commute_ferry',
            'One_met_Tram_or_lt_rail_P': 'commute_tram_lightrail',
            'One_method_Car_as_driver_P': 'commute_car_driver',
            'One_method_Walked_only_P': 'commute_walked',
            'Worked_home_P': 'worked_at_home',
            'geometry': 'geometry'
        }

        missing_cols = [col for col in columns_to_keep if col not in gdf_g62_raw.columns]
        if missing_cols:
            print(f"❌ Error: The following expected columns were not found in the raw data: {missing_cols}")
        else:
            gdf_g62 = gdf_g62_raw[list(columns_to_keep.keys())].copy()
            gdf_g62.rename(columns=columns_to_keep, inplace=True)
            print("Columns selected and renamed successfully.")

            print("\nCalculating combined public transport figure...")
            public_transport_cols = ['commute_train', 'commute_bus', 'commute_ferry', 'commute_tram_lightrail']
            gdf_g62['commute_public_transport_total'] = gdf_g62[public_transport_cols].sum(axis=1)
            print("Combined public transport calculated.")

            cols_order = ['sa1_code', 'total_employed_persons',
                          'commute_train', 'commute_bus', 'commute_ferry', 'commute_tram_lightrail',
                          'commute_public_transport_total', # Put combined here
                          'commute_car_driver', 'commute_walked', 'worked_at_home',
                          'geometry']
            gdf_g62 = gdf_g62[cols_order]


            print(f"Cleaned data shape: {gdf_g62.shape}")

            pd.set_option('display.max_rows', 10)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            print("\nCleaned Data Preview (G62 - Commute):")
            print(gdf_g62.head())

            print("\nBasic stats for numerical columns:")
            print(gdf_g62.describe(include=['number']))

            print(f"\nCoordinate Reference System (CRS): {gdf_g62.crs}")

            output_filename = "cleaned_data/cleaned_commute_sa62.gpkg"
            output_path = os.path.join(os.getcwd(), output_filename)
            print(f"\nSaving cleaned data to: {output_path}")
            try:
                gdf_g62.to_file(output_path, driver="GPKG", layer="Cleaned_data")
                print("--- Cleaned data saved successfully! ---💾")
            except Exception as e:
                print(f"❌ Error saving file: {e}")

    except Exception as e:
        print(f"\n❌ Error encountered during processing: {e}")

else:
    print(f"\n❌ Error: Cannot find the file at the specified path.")
    print(f"Path used: {file_path_g62}")