import logging
from typing import Any, Callable, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ErrorHandlingLoggingManager:
    """
    A class to handle errors and log messages during Hive's self-evolution process.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_message(self, message: str, level: str = 'INFO') -> None:
        """
        Logs a message with a specified level.

        :param message: The message to log.
        :param level: The logging level (INFO, ERROR, WARNING, etc.).
        """
        try:
            if level.upper() == 'INFO':
                self.logger.info(message)
            elif level.upper() == 'ERROR':
                self.logger.error(message)
            elif level.upper() == 'WARNING':
                self.logger.warning(message)
            elif level.upper() == 'CRITICAL':
                self.logger.critical(message)
            else:
                raise ValueError(f"Invalid log level: {level}")
        except Exception as e:
            self.handle_error(e)

    def handle_error(self, error: Exception) -> None:
        """
        Handles errors by logging them with an ERROR level.

        :param error: The error to handle.
        """
        self.log_message(str(error), 'ERROR')

    def wrap_with_logging(self, func: Callable) -> Callable:
        """
        Wraps a function with logging and error handling.

        :param func: The function to wrap.
        :return: The wrapped function.
        """
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle_error(e)
        return wrapper

# Unit tests for the ErrorHandlingLoggingManager class
def test_log_message():
    manager = ErrorHandlingLoggingManager()
    manager.log_message("Test log message", "INFO")
    assert "Test log message" in str(manager.logger.handlers[0].buffer)

def test_handle_error():
    manager = ErrorHandlingLoggingManager()
    mock_error = Exception("Mock error")
    with mock.patch.object(manager.logger, 'error') as mock_error_logger:
        manager.handle_error(mock_error)
        mock_error_logger.assert_called_once_with("Mock error")

def test_wrap_with_logging():
    manager = ErrorHandlingLoggingManager()
    @manager.wrap_with_logging
    def test_function():
        raise ValueError("Test exception")

    with mock.patch.object(manager.logger, 'error') as mock_error_logger:
        test_function()
        mock_error_logger.assert_called_once_with("Test exception")

# Run unit tests
if __name__ == '__main__':
    test_log_message()
    test_handle_error()
    test_wrap_with_logging()