
# Prerequisties

 - [Python](https://www.python.org/)
 - Stripe API key for Stripe Payment Integration
 - [Stripe webhook setup](https://stripe.com/docs/payments/handling-payment-events#install-cli)

# Installation
```
pip install -r requirements_fakeStore.txt
```




change pathfile of metadata.json in init.py accoridngly


create a .env 
use my env file given in environment example for now to avoid stripe integration and all

use the same square images dataset 
run dataset.py but dont store the json and just use the given metadata json and file to id json, make sure the image paths are like this only flask-O-Shop/app/static/uploads/dataset/{id}.jpg

run app.py it should be working now 

for the integration part
in app.py of the main site (not marketplace) change the json path files accordingly, first json is filename-id second is metadata
