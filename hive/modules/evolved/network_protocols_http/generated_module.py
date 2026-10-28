import unittest
from typing import List

class NetworkProtocolsHttpManager:
    """Manages HTTP network protocols with methods to handle common operations."""

    def __init__(self):
        """Initialize the HTTP manager."""
        pass

    def get_response(self, url: str) -> str:
        """
        Fetch the HTTP response from the given URL.

        Args:
            url (str): The URL to fetch the response from.

        Returns:
            str: The HTTP response content.

        Raises:
            ValueError: If the URL is not valid.
            Exception: For any other error that occurs during the request.
        """
        if not url:
            raise ValueError("URL cannot be empty.")
        # Simulate an HTTP GET request
        response = f"HTTP Response from {url}"
        return response

    def post_data(self, url: str, data: dict) -> str:
        """
        Send POST data to the given URL.

        Args:
            url (str): The URL to send the data to.
            data (dict): The data to send as part of the POST request.

        Returns:
            str: The HTTP response content.

        Raises:
            ValueError: If the URL or data is not valid.
            Exception: For any other error that occurs during the request.
        """
        if not url:
            raise ValueError("URL cannot be empty.")
        if not data:
            raise ValueError("Data cannot be empty.")
        # Simulate an HTTP POST request
        response = f"HTTP Response from {url} with data {data}"
        return response

    def delete_resource(self, url: str) -> str:
        """
        Delete a resource at the given URL.

        Args:
            url (str): The URL of the resource to delete.

        Returns:
            str: The HTTP response content.

        Raises:
            ValueError: If the URL is not valid.
            Exception: For any other error that occurs during the request.
        """
        if not url:
            raise ValueError("URL cannot be empty.")
        # Simulate an HTTP DELETE request
        response = f"HTTP Response from {url} (deleted)"
        return response


class TestNetworkProtocolsHttpManager(unittest.TestCase):
    """Unit tests for the NetworkProtocolsHttpManager class."""

    def setUp(self):
        """Set up the test environment."""
        self.manager = NetworkProtocolsHttpManager()

    def test_get_response(self):
        """Test the get_response method."""
        url = "http://example.com"
        response = self.manager.get_response(url)
        self.assertIn("HTTP Response from http://example.com", response)

    def test_post_data(self):
        """Test the post_data method."""
        url = "http://example.com"
        data = {"key": "value"}
        response = self.manager.post_data(url, data)
        self.assertIn("HTTP Response from http://example.com", response)
        self.assertIn("key=value", response)

    def test_delete_resource(self):
        """Test the delete_resource method."""
        url = "http://example.com/resource"
        response = self.manager.delete_resource(url)
        self.assertIn("HTTP Response from http://example.com/resource", response)
        self.assertIn("(deleted)", response)

if __name__ == "__main__":
    unittest.main()