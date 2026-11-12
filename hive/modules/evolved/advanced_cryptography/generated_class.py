import hashlib
from typing import Any


class AdvancedCryptographyManager:
    """
    A class to manage advanced cryptography functionalities.

    Methods:
        hash_data: Hashes the input data using SHA-256.
        encrypt_data: Encrypts the input data using AES encryption.
        decrypt_data: Decrypts the input data using AES encryption.
    """

    def __init__(self, key: bytes, iv: bytes):
        """
        Initializes the AdvancedCryptographyManager with a key and an initialization vector.

        :param key: The encryption key, should be 16, 24, or 32 bytes long.
        :param iv: The initialization vector, should be 16 bytes long.
        """
        self.key = key
        self.iv = iv
        from Crypto.Cipher import AES

        self.cipher = AES.new(self.key, AES.MODE_CBC, self.iv)

    def hash_data(self, data: Any) -> bytes:
        """
        Hashes the input data using SHA-256.

        :param data: The data to be hashed.
        :return: The SHA-256 hash of the data.
        """
        if not isinstance(data, (str, bytes)):
            raise TypeError("Data must be a string or bytes.")
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypts the input data using AES encryption.

        :param data: The data to be encrypted.
        :return: The encrypted data.
        """
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes.")
        # The padding is handled by the AES encryption
        return self.cipher.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts the input data using AES encryption.

        :param encrypted_data: The data to be decrypted.
        :return: The decrypted data.
        """
        if not isinstance(encrypted_data, bytes):
            raise TypeError("Encrypted data must be bytes.")
        from Crypto.Cipher import AES

        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        return cipher.decrypt(encrypted_data)


# Unit tests for each method
def test_hash_data():
    manager = AdvancedCryptographyManager(b"key", b"iv")
    assert manager.hash_data("hello").startswith("5e884898da28047151d0e56f8dc62927")


def test_encrypt_data():
    manager = AdvancedCryptographyManager(b"key", b"iv")
    data = b"hello"
    encrypted = manager.encrypt_data(data)
    assert len(encrypted) == len(data) + 16  # Padding added


def test_decrypt_data():
    manager = AdvancedCryptographyManager(b"key", b"iv")
    data = b"hello"
    encrypted = manager.encrypt_data(data)
    decrypted = manager.decrypt_data(encrypted)
    assert decrypted == data


# Run unit tests
test_hash_data()
test_encrypt_data()
test_decrypt_data()
