import osmium
import sys
import os

print("Starting tag scanning process...")

PBF_FILE_PATH = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/data/australia-251105.osm.pbf"
OUTPUT_DIR = "/Users/Zhuanz/Documents/ACADEMIC/fifth-semaster/Introduction-to-AI/final_project/script_to_clean_data/list_of_header"
OUTPUT_TXT_PATH = os.path.join(OUTPUT_DIR, "tag_scan_results.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# osmium to get ABS and ONLY data
class TagScannerHandler(osmium.SimpleHandler):
    def __init__(self):
        super(TagScannerHandler, self).__init__()
        self.amenity_values = set()
        self.shop_values = set()
        self.elements_processed = 0

    def process_tags(self, tags):
        if 'amenity' in tags:
            self.amenity_values.add(tags['amenity'])

        if 'shop' in tags:
            self.shop_values.add(tags['shop'])

        self.elements_processed += 1
        if self.elements_processed % 1000000 == 0:
            print(f"Processed {self.elements_processed // 1000000}M elements... "
                  f"(Found {len(self.amenity_values)} amenities, {len(self.shop_values)} shops)", end='\r')


    def node(self, n):
        self.process_tags(n.tags)

    def way(self, w):
        self.process_tags(w.tags)

    def relation(self, r):
        self.process_tags(r.tags)


print(f"Scanning PBF file: {PBF_FILE_PATH}")
print(f"Results will be saved to: {OUTPUT_TXT_PATH}")
print("This will take several minutes...")

if not os.path.exists(PBF_FILE_PATH):
    print(f"Error: Input file not found at {PBF_FILE_PATH}")
    sys.exit()

handler = TagScannerHandler()

try:
    handler.apply_file(PBF_FILE_PATH, locations=False)

finally:
    print("\n\nScan complete. Writing results to file...")

    try:
        with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write("--- Unique 'amenity' Tag Values Found ---\n")
            f.write("=" * 40 + "\n")
            # sort the result for better read perpose
            for amenity in sorted(handler.amenity_values):
                f.write(f"{amenity}\n")

            f.write("\n\n")
            f.write("--- Unique 'shop' Tag Values Found ---\n")
            f.write("=" * 40 + "\n")
            for shop in sorted(handler.shop_values):
                f.write(f"{shop}\n")

        print(f"Success! Results saved to: {OUTPUT_TXT_PATH}")
        print(f"Total unique 'amenity' values: {len(handler.amenity_values)}")
        print(f"Total unique 'shop' values: {len(handler.shop_values)}")

    except Exception as e:
        print(f"Error writing to output file: {e}")