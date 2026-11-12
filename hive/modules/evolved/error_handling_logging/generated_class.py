import logging
from typing import Any, Optional


class HiveEvolutionManager:
    """A class for managing error handling and logging functionalities in the Hive's self-evolution process."""

    def __init__(self, log_level: Optional[str] = "INFO"):
        """Initialize the logger with the specified level."""
        self.logger = logging.getLogger("HiveEvolutionManager")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def log_error(self, message: str, error: Exception):
        """Log an error message with the provided exception."""
        self.logger.error(f"{message} - {str(error)}")

    def handle_exception(self, function: callable, *args, **kwargs):
        """Handle exceptions that may occur during the execution of a function."""
        try:
            return function(*args, **kwargs)
        except Exception as e:
            self.log_error("An error occurred during the Hive's self-evolution process.", e)
            raise


# Unit tests for the HiveEvolutionManager class
if __name__ == "__main__":
    import unittest

    class TestHiveEvolutionManager(unittest.TestCase):
        def test_log_info(self):
            manager = HiveEvolutionManager()
            with unittest.mock.patch.object(manager.logger, "info") as mocked_info:
                manager.log_info("This is an info message")
                mocked_info.assert_called_once_with("This is an info message")

        def test_log_error(self):
            manager = HiveEvolutionManager()
            error_message = "This is an error message"
            exception = Exception("Test exception")
            with unittest.mock.patch.object(manager.logger, "error") as mocked_error:
                manager.log_error(error_message, exception)
                mocked_error.assert_called_once_with(f"{error_message} - {str(exception)}")

        def test_handle_exception(self):
            manager = HiveEvolutionManager()

            def test_function():
                raise Exception("Test exception")

            with self.assertRaises(Exception) as context:
                manager.handle_exception(test_function)
            self.assertEqual(str(context.exception), "Test exception")
            with unittest.mock.patch.object(manager.logger, "error") as mocked_error:
                manager.handle_exception(test_function)
                mocked_error.assert_called_once()
