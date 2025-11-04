import geopandas as  gpd

gpkg_file_path = '/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/data/Geopackage_2021_G33_NSW_GDA2020/G33_NSW_GDA2020.gpkg'
DATA_LAYER_NAME = 'G33_SA1_2021_NSW'

print(f"--- Start exploring file: {gpkg_file_path} ---")
print(f"--- Target Layer: {DATA_LAYER_NAME} ---")

try:
    gdf = gpd.read_file(gpkg_file_path,layer=DATA_LAYER_NAME,engine='pyogrio')

    all_columns = gdf.columns.tolist()
    print(f"\nFile Read Successful! Total Columns: {len(all_columns)} ---")

    print("\nColumn Headers (Row Names) List:")
    for col in all_columns:
        print(f" - {col}")
    print("\n-------------------")

    #geographic data
    id_col = next((col for col in all_columns if 'SA1_CODE' in col), 'SA1_ID_NOT_FOUND')
    print(f"Found Geographic ID Header: {id_col}")

    preview_cols = [id_col, 'geometry'] + [c for c in all_columns if c not in [id_col, 'geometry']][:4]
    print("\nPreview of First 5 Rows (ID, Geometry, and First 4 Data Cols):")
    print(gdf[preview_cols].head())


except:
    print(f"--- Error: Cannot read file: {gpkg_file_path} ---")
