# main.py
from src.core.service_runner import ServiceRunner

if __name__ == "__main__":
    service = ServiceRunner()
    service.start_service()
