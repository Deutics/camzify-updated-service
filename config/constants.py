"""
VAF Multistream Processing - Constants and Environment Variables
"""

from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()
load_dotenv(env_path)

# Twilio SMS Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_PHONE = os.getenv("TWILIO_FROM_PHONE")
TO_PHONE_NUMBERS = os.getenv("TO_PHONE_NUMBERS")

# Webhook Configuration
VAF_WEBHOOK_URL = os.getenv("VAF_WEBHOOK_URL")  # Different from fall detection webhook

# Location identifier
LOCATION = os.getenv("LOCATION", "55CPW")
