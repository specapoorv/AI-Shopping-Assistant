import json
from app import db
from app.db_models import Item

with open('/home/specapoorv/Downloads/Serial_comms_Kalman-20250522T115436Z-1-001/AI-Shopping-Assistant/fakeStore/static/uploads/metadata.json') as f:
    data = json.load(f)

for entry in data:
    item = Item(
        id=entry['id'],
        brand=entry['brand'],
        category=entry['category'],
        subcategory=entry.get('subcategory'),
        price_cents=entry['price_cents'],
        description=entry.get('description', ''),
        image_path=entry['image_path']
    )
    db.session.add(item)

db.session.commit()
print("Items added successfully.")