
# Prerequisties

 - [Python](https://www.python.org/)
 - Stripe API key for Stripe Payment Integration
 - [Stripe webhook setup](https://stripe.com/docs/payments/handling-payment-events#install-cli)

# Installation
```
pip install -r requirements_fakeStore.txt
```

change pathfile of metadata.json in init.py
create a .env 
use my env file for now to avoid stripe integration and all

use the same square images dataset 
run dataset.py but dont store the json and just use the given metadata json, make sure the image paths are like this only /static/uploads/dataset/{id}.jpg
