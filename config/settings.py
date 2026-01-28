import os
from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.camzify.com")
API_TOKEN = os.getenv("API_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n_custom.pt")

if not API_TOKEN:
   raise ValueError("Missing API_TOKEN in .env file")