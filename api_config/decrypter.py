import base64
import socket
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import unpad

import getmac
import uuid
from datetime import datetime


def decrypt_data(encrypted_data, aes_key="C8620628BE2507E2"):
    aes_key = b"C8620628BE2507E2"
    # encrypted_data = encrypted_data.encode('utf-16')
    delimiter = ":::"
    cipher_text, iv = encrypted_data.split(delimiter)
    # Decode the Base64-encoded data to get the raw bytes (IV + ciphertext)
    decoded_data = base64.b64decode(cipher_text)
    # Assuming the first 16 bytes of `decoded_data` are the Initialization Vector (IV)
    iv = base64.b64decode(iv)
    # The rest of the `decoded_data` is the ciphertext
    # ciphertext = decoded_data[16:]
    ciphertext = decoded_data
    # Create a new AES cipher object for decryption using the same key and IV
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    # Decrypt the ciphertext
    decrypted_data = cipher.decrypt(ciphertext)
    # Unpad the decrypted data (AES CBC requires padding)
    decrypted_data = unpad(decrypted_data, AES.block_size)
    # Convert the decrypted bytes back to a string (if it was originally a string)
    # print(decrypted_data.decode('utf-8'))
    # return decrypted_data.decode('utf-8')
    return decrypted_data.decode("utf-8")
