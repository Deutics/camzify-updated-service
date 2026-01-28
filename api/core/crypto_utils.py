import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import unpad
from typing import Optional
from utils_main.logger import get_logger

logger = get_logger(__name__)


def decrypt_aes_cbc(
    encrypted_data: str,
    aes_key: str = "C8620628BE2507E2",
    delimiter: str = ":::"
) -> Optional[str]:
    try:
        cipher_text, iv_encoded = encrypted_data.split(delimiter)
        
        decoded_data = base64.b64decode(cipher_text)
        iv = base64.b64decode(iv_encoded)
        
        key = aes_key.encode() if isinstance(aes_key, str) else aes_key
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(decoded_data)
        decrypted_data = unpad(decrypted_data, AES.block_size)
        
        return decrypted_data.decode("utf-8")
    
    except Exception as e:
        logger.error(f"AES decryption failed: {e}", exc_info=True)
        return None
