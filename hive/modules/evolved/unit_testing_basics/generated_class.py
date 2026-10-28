import unittest

class UnitTestingBasicsManager(unittest.TestCase):
    """
    A class that encapsulates unit testing fundamentals with methods for setup, test execution, and teardown.
    """

    def setUp(self):
        """
        Sets up the necessary environment for each test case.
        """
        self.test_data = {'key': 'value'}

    def test_data_structure(self):
        """
        Tests the structure of the test data.
        """
        self.assertIsInstance(self.test_data, dict)

    def test_key_value_pair(self):
        """
        Tests if the test data contains a specific key-value pair.
        """
        self.assertEqual(self.test_data['key'], 'value')

    def tearDown(self):
        """
        Cleans up the environment after each test case.
        """
        del self.test_data

if __name__ == '__main__':
    unittest.main()