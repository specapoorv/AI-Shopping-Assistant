import os
import shutil
import json
import random

source_folder = "/home/specapoorv/Downloads/Serial_comms_Kalman-20250522T115436Z-1-001/AI-Shopping-Assistant/ut-zap50k-images-square"
destination_folder = "./static/uploads/dataset"
os.makedirs(destination_folder, exist_ok=True)

metadata = []
filename_to_id = {}
current_id = 1

for root, dirs, files in os.walk(source_folder):
    for file in sorted(files):  # sorting ensures deterministic order
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            category = os.path.basename(os.path.dirname(os.path.dirname(root)))
            subcategory = os.path.basename(os.path.dirname(root))
            brand = os.path.basename(root)

            new_filename = f"{current_id}.jpg"
            src_path = os.path.join(root, file)
            dst_path = os.path.join(destination_folder, new_filename)

            try:
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")
                continue

            metadata.append({
                "id": current_id,
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
                "price_cents": random.randint(1000, 10000),
                "description": f"{brand} stylish {subcategory.lower()} from our {category.lower()} range.",
                "image_path": f"/static/uploads/dataset/{new_filename}"
            })

            filename_to_id[file] = current_id
            current_id += 1

# Save metadata
#with open("static/uploads/metadata.json", "w") as f:
 #   json.dump(metadata, f, indent=4)

# Save filename → id mapping
with open("filename_to_id.json", "w") as f:
    json.dump(filename_to_id, f, indent=2)

print(f"✅ Processed {len(metadata)} products and created filename_to_id.json")
