import unittest
from typing import Dict, List


class MachineLearningOpsManager:
    """A class for managing machine learning operations including training, validating, and testing models."""

    def __init__(self):
        """Initialize the MachineLearningOpsManager with empty dictionaries for storing models, data, and results."""
        self.models = {}
        self.data = {}
        self.results = {}

    def train_model(self, model_name: str, data_key: str) -> None:
        """
        Train a machine learning model using the data associated with the given data key.

        Args:
            model_name (str): The name of the model to train.
            data_key (str): The key to access the data required for training the model.

        Raises:
            ValueError: If the model or data is not found.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        if data_key not in self.data:
            raise ValueError(f"Data with key '{data_key}' not found.")

        model = self.models[model_name]
        data = self.data[data_key]

        # Simulate training process
        self.results[model_name] = model.train(data)

    def validate_model(self, model_name: str, validation_data_key: str) -> float:
        """
        Validate the machine learning model using the validation data associated with the given data key.

        Args:
            model_name (str): The name of the model to validate.
            validation_data_key (str): The key to access the validation data required for validating the model.

        Returns:
            float: The validation score of the model.

        Raises:
            ValueError: If the model or validation data is not found.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        if validation_data_key not in self.data:
            raise ValueError(f"Validation data with key '{validation_data_key}' not found.")

        model = self.models[model_name]
        validation_data = self.data[validation_data_key]

        # Simulate validation process
        validation_score = model.validate(validation_data)
        self.results[model_name] = validation_score
        return validation_score

    def test_model(self, model_name: str, test_data_key: str) -> float:
        """
        Test the machine learning model using the test data associated with the given data key.

        Args:
            model_name (str): The name of the model to test.
            test_data_key (str): The key to access the test data required for testing the model.

        Returns:
            float: The test score of the model.

        Raises:
            ValueError: If the model or test data is not found.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        if test_data_key not in self.data:
            raise ValueError(f"Test data with key '{test_data_key}' not found.")

        model = self.models[model_name]
        test_data = self.data[test_data_key]

        # Simulate testing process
        test_score = model.test(test_data)
        self.results[model_name] = test_score
        return test_score


# Unit tests for MachineLearningOpsManager
class TestMachineLearningOpsManager(unittest.TestCase):
    def setUp(self):
        self.manager = MachineLearningOpsManager()
        self.manager.models = {"model1": MockModel()}
        self.manager.data = {"train_data": [], "validation_data": [], "test_data": []}

    def test_train_model(self):
        self.manager.train_model("model1", "train_data")
        self.assertIn("model1", self.manager.results)

    def test_validate_model(self):
        validation_score = self.manager.validate_model("model1", "validation_data")
        self.assertIn("model1", self.manager.results)
        self.assertIsInstance(validation_score, float)

    def test_test_model(self):
        test_score = self.manager.test_model("model1", "test_data")
        self.assertIn("model1", self.manager.results)
        self.assertIsInstance(test_score, float)


# MockModel class for testing purposes
class MockModel:
    def train(self, data: List[Dict]) -> float:
        return 0.95  # Example training score

    def validate(self, validation_data: List[Dict]) -> float:
        return 0.96  # Example validation score

    def test(self, test_data: List[Dict]) -> float:
        return 0.97  # Example test score


# Run unit tests
if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
