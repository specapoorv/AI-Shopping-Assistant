from google import genai
from google.genai import types
from PIL import Image
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()


class MyLLMClass():
	def __init__(self):
		self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
		self.chat = None
		self.chat_msg_count = None
		self.clipcategory = None

	def get_single_llm_response(self, query, image=None):
		if image is None:
			response = self.client.models.generate_content(model="gemini-2.0-flash", contents=query)
		else:
			response = self.client.models.generate_content(model="gemini-2.0-flash", contents=[image, query])
		return response.text

	def create_new_chat(self):
		self.chat = self.client.chats.create(model="gemini-2.0-flash")
		self.chat_msg_count = 0

	def query_chat(self, query_text, image=None):
		if image is None:
			response = self.chat.send_message(query_text)
		else:
			# First query sends the beginning llm prompt
			response = self.chat.send_message([image, types.Part(text=query_text)])
		self.chat_msg_count += 1
		return response.text

	def extract_data_from_followup_responses(self, text):
		category, query, pricerange, brand, msg = text.strip().splitlines()[:5]
		category = category.split("CATEGORY:")[1].strip()
		query = query.split("QUERY:")[1].strip()
		pricerange = pricerange.split("BUDGET:")[1].strip()
		brand = brand.split("BRAND:")[1].strip()
		msg = msg.split("MESSAGE:")[1].strip()

		category = category.replace('_','\\')

		if pricerange.lower() == 'none':
			pricerange = None
		else:
			# Handling various formats of responses from the LLM or user
			a,b = pricerange.split(';')
			a = ''.join([ch for ch in a if ch.isnumeric()])
			b = ''.join([ch for ch in b if ch.isnumeric()])
			pricerange = (int(a.strip()), int(b.strip()))
		
		if brand.lower() == 'none':
			brand = None
		else:
			brand = brand.lower()

		return category, query, pricerange, brand, msg



beginning_llm_prompt = """You are a friendly personal shopping assistant. You have just been shown an image of a product the user likes. 

1. Start by briefly describing what you see in the image (e.g. category, color, key style elements).  
2. Then invite the user to share any extra preferences or context so you can refine your suggestions.  
3. Ask about things like their preferred styles, colors, sizes, brands, occasions, or budget limits.  
4. Keep your tone warm, conversational, and helpful—imagine you're a personal stylist in a boutique.  

Example output:
“Nice choice! I see a pair of black leather ankle boots with a pointed toe and block heel. Do you have a favorite brand or heel height? Or perhaps a color or material you’d like to explore next? Let me know your budget and any style details (like casual vs. dressy), and I'll pull up some perfect matches for you!”

"""

second_llm_prompt = """
For the next incoming messages, you have to now follow these instructions:
Your job on every user message:

1. Figure out which ONE of the following category strings best matches what the user now wants.  
(If they haven't clearly switched, keep the category you last output.)
Boots_Ankle
Boots_Knee High
Boots_Mid-Calf
Boots_Over the Knee
Boots_Prewalker Boots
Sandals_Athletic
Sandals_Flat
Sandals_Heel
Shoes_Boat Shoes
Shoes_Clogs and Mules
Shoes_Crib Shoes
Shoes_Firstwalker
Shoes_Flats
Shoes_Heels
Shoes_Loafers
Shoes_Oxfords
Shoes_Prewalker
Shoes_Sneakers and Athletic Shoes
Slippers_Boot
Slippers_Slipper Flats
Slippers_Slipper Heels


2. Write a concise, embed-ready description of the product(s) the user wants (colour, material, heel height, brand, occasion, budget, etc.).

3. **Return exactly two lines**—nothing else, no markdown:
CATEGORY: <one category from the list>
QUERY: <search description in 1-3 sentences>
BUDGET: <semicolon separated values for upper and lower bounds, both in Indian Rupees (1 us dollar = 85 indian rupees). If not specified, simply output None>
BRAND: <brand name. If bot specified, simply output None>
MESSAGE: <what you, as a friendly personal shopping assistant, would say after presenting the new set of choices. Invite them to ask further follow-up questions.>

Here is an example conversation (Note, I have not filled in the MESSAGE part. However, you, as a friendly shopping assistant, should fill it in as needed.)
As a personal shopping assistant, you will remember key details about the user's past messages.

User: “Do you have these boots in tan?” →  
CATEGORY: Boots_Ankle
QUERY: tan leather ankle boots with block heel
BUDGET: None
BRAND: None
MESSAGE: <insert message>

User: “Actually show me some flat strappy sandals under 1000” →  
CATEGORY: Sandals_Flat
QUERY: white or nude flat strappy sandals
BUDGET: 0; 1000
BRAND: None
MESSAGE: <insert message>

User: “Do you have adidas?” →  
CATEGORY: Sandals_Flat
QUERY: white or nude flat strappy sandals
BUDGET: 0; 1000
BRAND: Adidas
MESSAGE: <insert message>

Never add other words, greetings, or JSON. Just the five lines.

Your First User Message:
"""

clip_category_prompt = "The last detected category up until this message was "