from google import genai
from PIL import Image
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_llm_response(query, image=None):
	if image is None:
		response = client.models.generate_content(model="gemini-2.0-flash", contents=query)
	else:
		response = client.models.generate_content(model="gemini-2.0-flash", contents=[image, query])
	return response.text