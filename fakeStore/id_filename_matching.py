import os
import json

source_folder = "/home/specapoorv/Downloads/Serial_comms_Kalman-20250522T115436Z-1-001/AI-Shopping-Assistant/shoes"
filename_to_id = {}
current_id = 1

for category in sorted(os.listdir(source_folder)):
    cat_path = os.path.join(source_folder, category)
    if not os.path.isdir(cat_path):
        continue

    for subcategory in sorted(os.listdir(cat_path)):
        subcat_path = os.path.join(cat_path, subcategory)
        if not os.path.isdir(subcat_path):
            continue

        for brand in sorted(os.listdir(subcat_path)):
            brand_path = os.path.join(subcat_path, brand)
            if not os.path.isdir(brand_path):
                continue

            for image_name in sorted(os.listdir(brand_path)):
                if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    filename_to_id[image_name] = current_id
                    current_id += 1

# Save
with open("id_filename_matching.json", "w") as f:
    json.dump(filename_to_id, f, indent=2)

print("✅ Created filename → id mapping for Dataset B")
