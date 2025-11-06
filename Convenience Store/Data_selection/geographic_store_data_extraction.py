import osmium
import csv
import sys
import os

print("Starting store extraction process (V2)...")

DATA_PATH = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/data"
OUTPUT_DIR = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/Convenience Store/Data_for_Conven"
OSM_FILE_PATH = os.path.join(DATA_PATH, 'australia-251105.osm.pbf')
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'convenience_stores_locations.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_SHOP_TAGS = {
    "convenience",
    "conveneince",
    "corner_store",
    "milkbar",
    "Milk_Bar",
    "general_store",
    "General Store",
    "kiosk"
}

# osmium handler
class StoreLocationHandler(osmium.SimpleHandler):
    def __init__(self, writer):
        super(StoreLocationHandler, self).__init__()
        self.writer = writer
        self.stores_found = 0
        self.processed_ids = set()

    def check_tags(self, element_id, tags):
        if element_id in self.processed_ids:
            return None

        #automation to compound tags
        if 'shop' in tags:
            shop_value = tags['shop']
            tag_parts = shop_value.split(';')
            for part in tag_parts:
                if part in TARGET_SHOP_TAGS:
                    return True

        return False

    def write_location(self, element_id, location):
        try:
            lon = location.lon
            lat = location.lat

            self.writer.writerow([element_id, lon, lat])
            self.stores_found += 1
            self.processed_ids.add(element_id)

            if self.stores_found % 100 == 0:
                print(f"Found {self.stores_found} stores so far...", end='\r')

        except osmium.InvalidLocationError:
            pass

    def node(self, n):
        if self.check_tags(n.id, n.tags):
            self.write_location(n.id, n.location)

    def way(self, w):
        pass

    def area(self, a, i):
        i = 0
        if self.check_tags(a.id, a.tags):
            i = i + 1
            try:
                center_location = a.envelope.center
                self.write_location(a.id, center_location)
            except osmium.InvalidLocationError:
                pass
            except AttributeError:
                print(f"Skipping area {a.id}, could not get center.")
                print(f"i = {i}")
                pass


print(f"Scanning OSM file: {OSM_FILE_PATH}")
print(f"Looking for {len(TARGET_SHOP_TAGS)} types of shop tags...")
print("This may take several minutes...")

if not os.path.exists(OSM_FILE_PATH):
    print(f"Error: Input file not found at {OSM_FILE_PATH}")
    print("Please check the file paths and names.")
    sys.exit()

try:
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['osm_element_id', 'longitude', 'latitude'])

        handler = StoreLocationHandler(csv_writer)
        handler.apply_file(OSM_FILE_PATH)

finally:
    if 'handler' in locals() and handler.stores_found > 0:
        print(f"\n\nScan complete!")
        print(f"Success! Total convenience stores found: {handler.stores_found}")
        print(f"Locations saved to: {OUTPUT_CSV_PATH}")
    else:
        print("\n\nScan complete, but 0 stores were found or an error occurred.")