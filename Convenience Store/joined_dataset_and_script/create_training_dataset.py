import pandas as pd
import geopandas as gpd
import os
import time

print("Starting final training dataset creation...")
start_time = time.time()

BASE_PATH = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/Convenience Store/Data_for_Conven/"
ORIGINAL_GPKG_PATH = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/data/Geopackage_2021_G01_NSW_GDA2020/G01_NSW_GDA2020.gpkg"
GPKG_LAYER_NAME = "G01_SA1_2021_NSW"

store_locations_csv = os.path.join(BASE_PATH, 'convenience_stores_locations.csv')
features_csv = os.path.join(BASE_PATH, 'MASTER_Convenience_Store_Dataset.csv')

#final cleaned dataset
FINAL_TRAINING_DATASET_PATH = os.path.join(BASE_PATH, 'FINAL_TRAINING_DATASET.csv')

print(f"Loading store locations from {store_locations_csv}...")
try:
    stores_df = pd.read_csv(store_locations_csv)
    #convert pandas dataframe to geodataframe
    stores_gdf = gpd.GeoDataFrame(
        stores_df,
        geometry=gpd.points_from_xy(stores_df.longitude, stores_df.latitude),
        crs="EPSG:4326"
    )
    print(f"Loaded {len(stores_gdf)} store locations.")
except FileNotFoundError:
    print(f"Error: Store locations file not found at {store_locations_csv}")
    exit()

print("Loading SA1 features and shapes...")
try:
    features_df = pd.read_csv(features_csv)
    print(f"Loaded {len(features_df)} SA1 feature rows.")

    print(f"Loading geometries from {ORIGINAL_GPKG_PATH}...")
    sa1_shapes_gdf = gpd.read_file(
        ORIGINAL_GPKG_PATH,
        layer=GPKG_LAYER_NAME,
        engine="fiona",
        usecols=['SA1_CODE_2021', 'geometry']
    )

    #fixed code here:
    print("Fixing data types for merge...")
    features_df['SA1_CODE_2021'] = features_df['SA1_CODE_2021'].astype(str)
    sa1_shapes_gdf['SA1_CODE_2021'] = sa1_shapes_gdf['SA1_CODE_2021'].astype(str)

    sa1_main_gdf = sa1_shapes_gdf.merge(
        features_df,
        on='SA1_CODE_2021',
        how='right'
    )
    sa1_main_gdf = gpd.GeoDataFrame(sa1_main_gdf, crs=sa1_shapes_gdf.crs)
    print(f"Master SA1 GeoDataFrame created with shape: {sa1_main_gdf.shape}")

    #make sure two geopanda's dataframe are same
    print(f"Projecting stores CRS to match SA1 CRS ({sa1_main_gdf.crs})...")
    stores_gdf = stores_gdf.to_crs(sa1_main_gdf.crs)

except FileNotFoundError:
    print(f"Error: One of the master files not found. Check paths.")
    exit()
except Exception as e:
    print(f"Error during shape/feature loading: {e}")
    print("Check your GPKG path and layer name in the script.")
    exit()

# Spatial Join
print("Performing spatial join... (This may take a moment)")
joined_gdf = gpd.sjoin(stores_gdf, sa1_main_gdf, how="inner", predicate="within")
print(f"Spatial join complete. Found {len(joined_gdf)} stores located within SA1 regions.")

print("Calculating target variable (Y) 'store_count'...")
if joined_gdf.empty:
    print("Warning: Spatial join resulted in 0 matches. No stores were found inside the provided SA1 regions.")
    sa1_main_gdf['store_count'] = 0
else:
    store_counts = joined_gdf.groupby('index_right').size().reset_index(name='store_count')
    store_counts_by_sa1 = joined_gdf.groupby('SA1_CODE_2021').size().reset_index(name='store_count')

    print("Merging 'store_count' back into master table...")
    final_gdf = sa1_main_gdf.merge(store_counts_by_sa1, on='SA1_CODE_2021', how='left')

    final_gdf['store_count'] = final_gdf['store_count'].fillna(0).astype(int)

print("Saving final training dataset...")

# remove the index 'shape' from dataset cause there is no need for training AI
final_df_to_save = pd.DataFrame(final_gdf.drop(columns='geometry'))

final_df_to_save.to_csv(FINAL_TRAINING_DATASET_PATH, index=False)

end_time = time.time()
print("\n--- SUCCESS! ---")
print(f"Final training dataset saved to: {FINAL_TRAINING_DATASET_PATH}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("\n--- Data Head (Top 5 rows) ---")
print(final_df_to_save.head())
print("\n--- 'store_count' 统计 ---")
print(final_df_to_save['store_count'].value_counts().sort_index().head(10))