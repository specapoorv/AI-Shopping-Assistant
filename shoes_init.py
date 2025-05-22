import os
import shutil
import json
from pathlib import Path

def flatten_brands(shoes_root):
    """
    shoes_root/
      └── <category>/
          └── <subcategory>/
              └── <brand>/
                  └── image.jpg
      
    After running, images live in:
      shoes_root/<category>/<subcategory>/<category>_<subcategory>_image.jpg
    and brand folders are removed.
    
    Returns a dict: { "category_subcategory_image.jpg": "brand", ... }
    """
    shoes_root = Path(shoes_root)
    mapping = {}

    # iterate categories
    for cat_dir in shoes_root.iterdir():
        if not cat_dir.is_dir():
            continue
        # iterate subcategories
        for sub_dir in cat_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            # iterate brands
            for brand_dir in sub_dir.iterdir():
                if not brand_dir.is_dir():
                    continue
                brand = brand_dir.name
                # move each image up one level
                for img_path in brand_dir.iterdir():
                    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    new_json_name = f"{cat_dir.name}_{sub_dir.name}_{img_path.name}"
                    new_name = f"{img_path.name}"
                    dest_path = sub_dir / new_name
                    # if collision, you can choose to overwrite or skip
                    if dest_path.exists():
                        print(f"⚠️ {dest_path.name} already exists—skipping")
                        continue
                    shutil.move(str(img_path), str(dest_path))
                    mapping[new_json_name] = brand

                # try to remove empty brand folder
                try:
                    brand_dir.rmdir()
                except OSError:
                    print(f"⚠️ Could not remove {brand_dir}, it may not be empty")

    return mapping

if __name__ == "__main__":
    shoes_dir = "shoes"      # ← change this to your actual path
    brand_map = flatten_brands(shoes_dir)

    # print out the mapping
    print("Mapping of new filenames → brand:")
    for fname, brand in brand_map.items():
        print(f"  {fname}  :  {brand}")

    # optionally save to JSON
    with open("brand_mapping.json", "w", encoding="utf-8") as f:
        json.dump(brand_map, f, indent=2, ensure_ascii=False)

    print("\nDone! All brand folders removed, images flattened, and mapping saved.")
