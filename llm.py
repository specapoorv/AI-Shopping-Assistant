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
		category, query, msg = text.strip().splitlines()[:3]
		category = category.split("CATEGORY: ")[1]
		query = query.split("QUERY: ")[1]
		msg = msg.split("MESSAGE: ")[1]
		category = category.replace('_','\\')
		return category, query, msg