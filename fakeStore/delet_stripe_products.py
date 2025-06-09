import stripe
import os
from dotenv import load_dotenv

load_dotenv()

def delete_all_prices_and_products():
    products = stripe.Product.list(limit=100)
    
    while True:
        for product in products.data:
            try:
                # List prices for the product
                prices = stripe.Price.list(product=product.id, limit=100)
                for price in prices.data:
                    try:
                        stripe.Price.delete(price.id)
                        print(f"Deleted price {price.id} for product {product.id}")
                    except Exception as e:
                        print(f"Failed to delete price {price.id}: {e}")
                
                # After deleting all prices, delete the product
                stripe.Product.delete(product.id)
                print(f"Deleted product {product.id}")
            except Exception as e:
                print(f"Failed to delete product {product.id}: {e}")

        if products.has_more:
            products = stripe.Product.list(limit=100, starting_after=products.data[-1].id)
        else:
            break

if __name__ == "__main__":
    delete_all_prices_and_products()
