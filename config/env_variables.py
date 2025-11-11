# config/env_variables.py
import os
from dotenv import load_dotenv

class EnvVariables:

    # Load from envs/.env relative to project root
    ENV_PATH = os.path.join(os.path.dirname(__file__), '..', 'envs', '.env')
    load_dotenv(ENV_PATH)

    BASE_URL = os.getenv("base_url")
    API_TOKEN = os.getenv("secret_key")
    MODEL_PATH = os.getenv("model_path", "ObjectDetector/yolov11/data_models/yolo11n_custom.pt")

    if not BASE_URL or not API_TOKEN:
        raise ValueError("Missing base_url or secret_key in envs/.env file.")
