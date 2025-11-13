import geopandas as gpd
import pandas as pd
import osmium
import os
import sys
import time

print("--- Starting OSM Feature Counter Script ---")
start_time = time.time()

BASE_DATA_PATH = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/data/"
OSM_FILE_PATH = os.path.join(BASE_DATA_PATH, 'australia-251105.osm.pbf')
GPKG_PATH = os.path.join(BASE_DATA_PATH, "Geopackage_2021_G01_NSW_GDA2020/G01_NSW_GDA2020.gpkg")
GPKG_LAYER_NAME = "G01_SA1_2021_NSW"

OUTPUT_DIR = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/Convenience_Store/Data_for_Conven"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'osm_features.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)


TARGET_FEATURES = {
    #Competition
    'competitor_supermarket_count': ('shop', 'supermarket'),
    'competitor_general_store_count': ('shop', 'general_store'),

    #Traffic Drivers
    'school_count': ('amenity', 'school'),
    'university_count': ('amenity', 'university'),
    'hospital_count': ('amenity', 'hospital'),
    'restaurant_count': ('amenity', 'restaurant'),
    'cafe_count': ('amenity', 'cafe'),
    'office_count': ('office', '*'),

    #Infrastructure & Accessibility
    'bus_stop_count': ('amenity', 'bus_stop'),
    'parking_count': ('amenity', 'parking'),
    'atm_count': ('amenity', 'atm'),
    'bank_count': ('amenity', 'bank'),
    'post_office_count': ('amenity', 'post_office')
}

TAG_FILTER = {}
for col_name, (key, value) in TARGET_FEATURES.items():
    if key not in TAG_FILTER:
        TAG_FILTER[key] = set()
    TAG_FILTER[key].add(value)


# Osmium processing unit
class FeatureLocationHandler(osmium.SimpleHandler):
    def __init__(self):
        super(FeatureLocationHandler, self).__init__()
        self.features_list = []
        self.reverse_lookup = {v: k for k, v in TARGET_FEATURES.items()}

    def check_element_tags(self, tags):
        for tag_key, tag_value in tags:
            if tag_key in TAG_FILTER:
                if '*' in TAG_FILTER[tag_key] or tag_value in TAG_FILTER[tag_key]:
                    col_name = self.reverse_lookup.get((tag_key, tag_value))
                    if not col_name:
                        col_name = self.reverse_lookup.get((tag_key, '*'))

                    if col_name:
                        return col_name
        return None

    def add_feature(self, col_name, location):
        try:
            self.features_list.append({
                'feature_type': col_name,
                'lon': location.lon,
                'lat': location.lat
            })
        except osmium.InvalidLocationError:
            pass

    def node(self, n):
        col_name = self.check_element_tags(n.tags)
        if col_name:
            self.add_feature(col_name, n.location)

    def area(self, a):
        col_name = self.check_element_tags(a.tags)
        if col_name:
            try:
                center_location = a.envelope.center
                self.add_feature(col_name, center_location)
            except Exception:
                pass


print(f"Step 1/5: Start scanning OSM file: {OSM_FILE_PATH}")
print(f"looking for {len(TARGET_FEATURES)} types of features...")

handler = FeatureLocationHandler()
handler.apply_file(OSM_FILE_PATH, locations=True)

print(f"\n    ...scan complete. extract from OSM {len(handler.features_list)} features")

if not handler.features_list:
    print("Error: Check TARGET_FEATURES list again.")
    sys.exit()

print(f"Step 2/5: loading SA1 area's shape: {GPKG_PATH}")
try:
    sa1_shapes_gdf = gpd.read_file(
        GPKG_PATH,
        layer=GPKG_LAYER_NAME,
        usecols=['SA1_CODE_2021', 'geometry']
    )
    sa1_shapes_gdf['SA1_CODE_2021'] = sa1_shapes_gdf['SA1_CODE_2021'].astype(str)
except Exception as e:
    print(f"Error: CANNOT LOAD GPKG file: {e}")
    sys.exit()

print("Step 3/5: converting OSM Features to GeoDataFrame...")
features_df = pd.DataFrame(handler.features_list)
features_gdf = gpd.GeoDataFrame(
    features_df,
    geometry=gpd.points_from_xy(features_df.lon, features_df.lat),
    crs="EPSG:4326"
)
print(f"    ...Convert process have finished CRS = {features_gdf.crs}")

# spatial connection
print(f"Step 4/5: Synchronise the data point and start the Spetial conection process (sjoin)...")
print(f"    ...project {len(features_gdf)} features to SA1's CRS ({sa1_shapes_gdf.crs})")

features_gdf = features_gdf.to_crs(sa1_shapes_gdf.crs)

# Apply the Spatial connection
joined_gdf = gpd.sjoin(
    features_gdf,
    sa1_shapes_gdf,
    how="inner",
    predicate="within"
)
print(f"    ...Spatial connection has finished {len(joined_gdf)} Features are matching to SA1 AREA.")


print("Step 5/5: Count based on SA1 area's Features counting...")

if joined_gdf.empty:
    print("Error: The result of spatial connection. your OSM points and GPKG areamay not overlayed")
    sys.exit()

counts = joined_gdf.groupby(['SA1_CODE_2021', 'feature_type']).size()

features_count_df = counts.unstack(level='feature_type', fill_value=0)
final_df = sa1_shapes_gdf[['SA1_CODE_2021']].merge(
    features_count_df,
    on='SA1_CODE_2021',
    how='left'
)

final_df = final_df.fillna(0)

count_columns = list(TARGET_FEATURES.keys())
existing_count_columns = [col for col in count_columns if col in final_df.columns]
final_df[existing_count_columns] = final_df[existing_count_columns].astype(int)

final_df.to_csv(OUTPUT_CSV_PATH, index=False)

end_time = time.time()
print("\nSUCESSFUL !!!!!")
print(f"New feature file has been save to: {OUTPUT_CSV_PATH}")
print(f"Total  cost: {end_time - start_time:.2f} second")
print("\nHead review")
print(final_df.head())