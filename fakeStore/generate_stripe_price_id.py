# generate_stripe_metadata.py
# no use of this instead stripe product is created after checkout see in init.py

import stripe
import json
import os
from dotenv import load_dotenv

load_dotenv()
stripe.api_key ="sk_test_51RXG0PP9PkOnkc6MbJedioezSElNtO506u2rjg59THZ2ELn7bE6zvtcClfVjhJC53nyHZbPWuNqfrFOboFQW3Xvm00mQUOTLGu"

INPUT_JSON = "/home/specapoorv/Downloads/Serial_comms_Kalman-20250522T115436Z-1-001/AI-Shopping-Assistant/fakeStore/static/uploads/metadata.json"
OUTPUT_JSON = "/home/specapoorv/Downloads/Serial_comms_Kalman-20250522T115436Z-1-001/AI-Shopping-Assistant/fakeStore/static/uploads/metadata_with_stripe.json"

with open(INPUT_JSON) as f:
    data = json.load(f)

updated_data = []
for item in data:
    try:
        # Skip invalid or missing prices
        if "price_cents" not in item or not isinstance(item["price_cents"], int) or item["price_cents"] <= 0:
            print(f"❌ Skipping ID {item['id']} due to invalid price: {item.get('price_cents')}")
            continue

        # Use id as the name to avoid duplicates
        product = stripe.Product.create(name=str(item["id"]))

        price = stripe.Price.create(
            unit_amount=item["price_cents"],
            currency="usd",
            product=product.id,
        )

        item["stripe_price_id"] = price.id
        updated_data.append(item)

        print(f"✅ Added Stripe price_id for ID {item['id']}: {price.id}")
    except Exception as e:
        print(f"❌ Failed to create price for ID {item['id']}: {e}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(updated_data, f, indent=2)

print(f"✅ New metadata saved to: {OUTPUT_JSON}")
