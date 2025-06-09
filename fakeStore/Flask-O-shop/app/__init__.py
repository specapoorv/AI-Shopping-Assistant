print("Loading Flask app __init__.py")




import os, stripe, json
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, flash, request, abort
from flask_bootstrap import Bootstrap
from .forms import LoginForm, RegisterForm
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, current_user, login_required, logout_user
from .db_models import db, User, Item
from itsdangerous import URLSafeTimedSerializer
from .funcs import mail, send_confirmation_email, fulfill_order
from dotenv import load_dotenv
from .admin.routes import admin


load_dotenv(dotenv_path= "/home/specapoorv/Downloads/Serial_comms_Kalman-20250522T115436Z-1-001/AI-Shopping-Assistant/fakeStore/Flask-O-shop/.env")
print("SECRET_KEY from env:", os.getenv("SECRET_KEY"))
print("DB_URI from env:", os.getenv("DB_URI"))


app = Flask(__name__)
app.register_blueprint(admin)

app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ["DB_URI"]
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_USERNAME'] = os.environ["EMAIL"]
app.config['MAIL_PASSWORD'] = os.environ["PASSWORD"]
app.config['MAIL_SERVER'] = "smtp.googlemail.com"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_PORT'] = 587
stripe.api_key = os.environ["STRIPE_PRIVATE"]

Bootstrap(app)
db.init_app(app)
mail.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

with app.app_context():
	db.create_all()

@app.context_processor
def inject_now():
	""" sends datetime to templates as 'now' """
	return {'now': datetime.utcnow()}

@login_manager.user_loader
def load_user(user_id):
	return User.query.get(user_id)

@app.route("/")
def home():
	items = Item.query.limit(20).all()
	return render_template("home.html", items=items)




@app.route("/login", methods=['POST', 'GET'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		email = form.email.data
		user = User.query.filter_by(email=email).first()
		if user == None:
			flash(f'User with email {email} doesn\'t exist!<br> <a href={url_for("register")}>Register now!</a>', 'error')
			return redirect(url_for('login'))
		elif check_password_hash(user.password, form.password.data):
			login_user(user)
			return redirect(url_for('home'))
		else:
			flash("Email and password incorrect!!", "error")
			return redirect(url_for('login'))
	return render_template("login.html", form=form)

@app.route("/register", methods=['POST', 'GET'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegisterForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		if user:
			flash(f"User with email {user.email} already exists!!<br> <a href={url_for('login')}>Login now!</a>", "error")
			return redirect(url_for('register'))
		new_user = User(name=form.name.data,
						email=form.email.data,
						password=generate_password_hash(
									form.password.data,
									method='pbkdf2:sha256',
									salt_length=8),
						phone=form.phone.data)
		db.session.add(new_user)
		db.session.commit()
		# send_confirmation_email(new_user.email)
		flash('Thanks for registering! You may login now.', 'success')
		return redirect(url_for('login'))
	return render_template("register.html", form=form)

@app.route('/confirm/<token>')
def confirm_email(token):
	try:
		confirm_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
		email = confirm_serializer.loads(token, salt='email-confirmation-salt', max_age=3600)
	except:
		flash('The confirmation link is invalid or has expired.', 'error')
		return redirect(url_for('login'))
	user = User.query.filter_by(email=email).first()
	if user.email_confirmed:
		flash(f'Account already confirmed. Please login.', 'success')
	else:
		user.email_confirmed = True
		db.session.add(user)
		db.session.commit()
		flash('Email address successfully confirmed!', 'success')
	return redirect(url_for('login'))

@app.route("/logout")
@login_required
def logout():
	logout_user()
	return redirect(url_for('login'))

@app.route("/resend")
@login_required
def resend():
	send_confirmation_email(current_user.email)
	logout_user()
	flash('Confirmation email sent successfully.', 'success')
	return redirect(url_for('login'))

@app.route("/add/<id>", methods=['POST'])
def add_to_cart(id):
	if not current_user.is_authenticated:
		flash(f'You must login first!<br> <a href={url_for("login")}>Login now!</a>', 'error')
		return redirect(url_for('login'))

	item = Item.query.get(id)
	if request.method == "POST":
		quantity = request.form["quantity"]
		current_user.add_to_cart(id, quantity)
		flash(f'''{item.brand} successfully added to the <a href=cart>cart</a>.<br> <a href={url_for("cart")}>view cart!</a>''','success')
		return redirect(url_for('home'))
	

@app.route("/cart")
@login_required
def cart():
	price = 0
	price_ids = []
	items = []
	quantity = []
	for cart in current_user.cart:
		items.append(cart.item)
		quantity.append(cart.quantity)
		price_id_dict = {
			"price": cart.item.price_id,
			"quantity": cart.quantity,
			}
		price_ids.append(price_id_dict)
		price += cart.item.price_cents*cart.quantity
	return render_template('cart.html', items=items, price=price, price_ids=price_ids, quantity=quantity)

@app.route('/orders')
@login_required
def orders():
	return render_template('orders.html', orders=current_user.orders)

@app.route("/remove/<id>/<quantity>")
@login_required
def remove(id, quantity):
	current_user.remove_from_cart(id, quantity)
	return redirect(url_for('cart'))

@app.route('/item/<int:id>')
def item(id):
	item = Item.query.get(id)
	return render_template('item.html', item=item)

@app.route('/search')
def search():
	query = request.args['query']
	search = "%{}%".format(query)
	items = Item.query.filter(Item.brand.like(search)).all()
	return render_template('home.html', items=items, search=True, query=query)

# stripe stuffs
@app.route('/payment_success')
def payment_success():
	return render_template('success.html')

@app.route('/payment_failure')
def payment_failure():
	return render_template('failure.html')

import stripe
import os

stripe.api_key ="sk_test_51RXG0PP9PkOnkc6MbJedioezSElNtO506u2rjg59THZ2ELn7bE6zvtcClfVjhJC53nyHZbPWuNqfrFOboFQW3Xvm00mQUOTLGu"

with open("/home/specapoorv/fakeStore/static/uploads/metadata.json") as f:
    metadata = json.load(f)

def get_item_info_by_id(item_id):
    for item in metadata:
        if str(item["id"]) == str(item_id):
            return {
                "id": item["id"],
                "price_cents": item["price_cents"]
            }
    return None

def get_or_create_stripe_price(item):
    existing_products = stripe.Product.list(limit=100, active=True)
    product = None
    for p in existing_products.auto_paging_iter():
        if p.name == str(item["id"]):
            product = p
            break

    if not product:
        product = stripe.Product.create(name=str(item["id"]))
        print(f"✅ Created product {product.id}")

    existing_prices = stripe.Price.list(product=product.id, limit=100)
    price = None
    for pr in existing_prices.auto_paging_iter():
        if pr.unit_amount == item["price_cents"]:
            price = pr
            break

    if not price:
        price = stripe.Price.create(
            unit_amount=item["price_cents"],
            currency="usd",
            product=product.id,
        )
        print(f"✅ Created price {price.id}")

    return price.id

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        items = json.loads(request.form['price_ids'].replace("'", '"'))  # list of {"id": ..., "quantity": ...}
        line_items = []

        for cart_item in items:
            item_id = cart_item["id"]
            item_data = get_item_info_by_id(item_id)

            if not item_data:
                return f" Product ID {item_id} not found in metadata", 400

            price_id = get_or_create_stripe_price(item_data)

            line_items.append({
                "price": price_id,
                "quantity": cart_item["quantity"]
            })

        checkout_session = stripe.checkout.Session.create(
            client_reference_id=current_user.id,
            line_items=line_items,
            payment_method_types=['card'],
            mode='payment',
            success_url=url_for('payment_success', _external=True),
            cancel_url=url_for('payment_failure', _external=True),
        )
        return redirect(checkout_session.url, code=303)

    except Exception as e:
        return f"Error creating checkout session: {e}", 500

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():

	if request.content_length > 1024*1024:
		print("Request too big!")
		abort(400)

	payload = request.get_data()
	sig_header = request.environ.get('HTTP_STRIPE_SIGNATURE')
	ENDPOINT_SECRET = os.environ.get('ENDPOINT_SECRET')
	event = None

	try:
		event = stripe.Webhook.construct_event(
		payload, sig_header, ENDPOINT_SECRET
		)
	except ValueError as e:
		# Invalid payload
		return {}, 400
	except stripe.error.SignatureVerificationError as e:
		# Invalid signature
		return {}, 400

	if event['type'] == 'checkout.session.completed':
		session = event['data']['object']

		# Fulfill the purchase...
		fulfill_order(session)

	# Passed signature verification
	return {}, 200
