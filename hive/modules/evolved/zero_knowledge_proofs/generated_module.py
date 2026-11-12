import hashlib
import json
import unittest
from typing import Any, Dict, List


class ZeroKnowledgeProofsManager:
    """
    ZeroKnowledgeProofsManager is a class to manage zero-knowledge proofs.
    """

    def __init__(self):
        """
        Initializes the ZeroKnowledgeProofsManager with an empty dictionary to store proofs.
        """
        self.proofs = {}

    def generate_proof(self, assertion: str, challenge: str) -> str:
        """
        Generates a zero-knowledge proof for a given assertion and challenge.

        Parameters:
        - assertion (str): The assertion that needs to be proven.
        - challenge (str): The challenge to be provided by the verifier.

        Returns:
        - str: The zero-knowledge proof.
        """
        try:
            proof = hashlib.sha256(f"{assertion}{challenge}".encode()).hexdigest()
            return proof
        except Exception as e:
            raise ValueError(f"Error generating proof: {e}")

    def verify_proof(self, assertion: str, challenge: str, proof: str) -> bool:
        """
        Verifies the zero-knowledge proof for a given assertion, challenge, and proof.

        Parameters:
        - assertion (str): The assertion that was proven.
        - challenge (str): The challenge provided during the proof generation.
        - proof (str): The zero-knowledge proof to be verified.

        Returns:
        - bool: True if the proof is valid, False otherwise.
        """
        try:
            generated_proof = self.generate_proof(assertion, challenge)
            return generated_proof == proof
        except Exception as e:
            raise ValueError(f"Error verifying proof: {e}")

    def store_proof(self, assertion: str, challenge: str, proof: str) -> None:
        """
        Stores a proof in the manager's dictionary for future reference.

        Parameters:
        - assertion (str): The assertion that was proven.
        - challenge (str): The challenge provided during the proof generation.
        - proof (str): The zero-knowledge proof.
        """
        try:
            self.proofs[(assertion, challenge)] = proof
        except Exception as e:
            raise ValueError(f"Error storing proof: {e}")

    def get_proof(self, assertion: str, challenge: str) -> str:
        """
        Retrieves a stored proof from the manager's dictionary.

        Parameters:
        - assertion (str): The assertion that was proven.
        - challenge (str): The challenge provided during the proof generation.

        Returns:
        - str: The stored zero-knowledge proof.
        """
        try:
            return self.proofs.get((assertion, challenge), None)
        except Exception as e:
            raise ValueError(f"Error retrieving proof: {e}")


class TestZeroKnowledgeProofsManager(unittest.TestCase):
    def setUp(self):
        self.manager = ZeroKnowledgeProofsManager()

    def test_generate_and_verify_proof(self):
        assertion = "The secret is 'password'"
        challenge = "Is this the real secret?"
        proof = self.manager.generate_proof(assertion, challenge)
        self.assertTrue(self.manager.verify_proof(assertion, challenge, proof))

    def test_store_and_retrieve_proof(self):
        assertion = "The number is 42"
        challenge = "Is this the answer to life?"
        proof = self.manager.generate_proof(assertion, challenge)
        self.manager.store_proof(assertion, challenge, proof)
        retrieved_proof = self.manager.get_proof(assertion, challenge)
        self.assertEqual(proof, retrieved_proof)

    def test_invalid_proof(self):
        assertion = "The secret is 'password'"
        challenge = "Is this the real secret?"
        proof = self.manager.generate_proof(assertion, challenge)
        invalid_proof = proof + "123"
        self.assertFalse(self.manager.verify_proof(assertion, challenge, invalid_proof))


if __name__ == "__main__":
    unittest.main()
