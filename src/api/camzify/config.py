class CamzifyConfig:
    LOGIN_ENDPOINT = "/api/v1/auth/login/"
    FEATURE_ENDPOINT_TEMPLATE = "/api/v1/instance/stream/analytic/{feature}/instance"
    
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_TIMEOUT = 15
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_BACKOFF = 0.5
    DEFAULT_SEMAPHORE_LIMIT = 10
    
    AES_KEY = "C8620628BE2507E2"
    AES_DELIMITER = ":::"