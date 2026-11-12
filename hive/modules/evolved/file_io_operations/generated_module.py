import os
import unittest


class FileIoOperationsManager:
    """
    A class for handling file input/output operations.
    """

    def __init__(self, file_path: str):
        """
        Initializes the FileIoOperationsManager with a file path.

        :param file_path: str - The path to the file to be managed.
        """
        self.file_path = file_path

    def read_file(self) -> str:
        """
        Reads the contents of the file.

        :return: str - The contents of the file.
        :raises FileNotFoundError: If the file does not exist.
        :raises IOError: If there is an error reading the file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        try:
            with open(self.file_path, "r") as file:
                return file.read()
        except IOError as e:
            raise IOError(f"An error occurred while reading the file: {e}")

    def write_file(self, content: str) -> None:
        """
        Writes the given content to the file.

        :param content: str - The content to write to the file.
        :raises IOError: If there is an error writing to the file.
        """
        try:
            with open(self.file_path, "w") as file:
                file.write(content)
        except IOError as e:
            raise IOError(f"An error occurred while writing to the file: {e}")

    def append_to_file(self, content: str) -> None:
        """
        Appends the given content to the end of the file.

        :param content: str - The content to append to the file.
        :raises IOError: If there is an error appending to the file.
        """
        try:
            with open(self.file_path, "a") as file:
                file.write(content)
        except IOError as e:
            raise IOError(f"An error occurred while appending to the file: {e}")


class TestFileIoOperationsManager(unittest.TestCase):
    """
    Unit tests for the FileIoOperationsManager class.
    """

    def setUp(self):
        self.test_file_path = "test_file.txt"
        self.manager = FileIoOperationsManager(self.test_file_path)

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_read_file(self):
        """
        Test the read_file method.
        """
        with open(self.test_file_path, "w") as file:
            file.write("Hello, World!")
        self.assertEqual(self.manager.read_file(), "Hello, World!")

    def test_write_file(self):
        """
        Test the write_file method.
        """
        content = "Test content"
        self.manager.write_file(content)
        with open(self.test_file_path, "r") as file:
            self.assertEqual(file.read(), content)

    def test_append_to_file(self):
        """
        Test the append_to_file method.
        """
        content = "First line"
        self.manager.write_file(content)
        content += "\nSecond line"
        self.manager.append_to_file(content)
        with open(self.test_file_path, "r") as file:
            self.assertEqual(file.read(), content)

    def test_file_not_found_error(self):
        """
        Test that FileNotFoundError is raised when file does not exist.
        """
        manager = FileIoOperationsManager("non_existent_file.txt")
        with self.assertRaises(FileNotFoundError):
            manager.read_file()

    def test_io_error(self):
        """
        Test that IOError is raised when an I/O error occurs.
        """
        with self.assertRaises(IOError):
            self.manager.write_file("This should raise an IOError")


if __name__ == "__main__":
    unittest.main()
