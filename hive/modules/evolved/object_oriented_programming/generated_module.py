import unittest


class ObjectOrientedProgrammingManager:
    """
    ObjectOrientedProgrammingManager is a class that demonstrates object-oriented programming principles.
    It includes methods to create objects, manipulate them, and perform operations.
    """

    def __init__(self):
        """
        Initializes the ObjectOrientedProgrammingManager with an empty list of objects.
        """
        self.objects = []

    def create_object(self, name, attributes):
        """
        Creates a new object with the given name and attributes and adds it to the list of objects.

        :param name: str - the name of the object
        :param attributes: dict - a dictionary of attributes for the object
        :return: None
        """
        new_object = {name: attributes}
        self.objects.append(new_object)

    def get_object(self, name):
        """
        Retrieves an object by its name from the list of objects.

        :param name: str - the name of the object to retrieve
        :return: dict - the object with the given name, or None if not found
        """
        for obj in self.objects:
            if name in obj:
                return obj[name]
        return None

    def update_object(self, name, new_attributes):
        """
        Updates the attributes of an object by its name.

        :param name: str - the name of the object to update
        :param new_attributes: dict - a dictionary of new attributes to add or update
        :return: None
        """
        obj = self.get_object(name)
        if obj:
            obj.update(new_attributes)

    def delete_object(self, name):
        """
        Deletes an object by its name from the list of objects.

        :param name: str - the name of the object to delete
        :return: None
        """
        self.objects = [obj for obj in self.objects if name not in obj]


class TestObjectOrientedProgrammingManager(unittest.TestCase):
    """
    Test cases for the ObjectOrientedProgrammingManager class.
    """

    def test_create_object(self):
        manager = ObjectOrientedProgrammingManager()
        manager.create_object("test_object", {"color": "red"})
        self.assertIn("test_object", manager.objects)
        self.assertEqual(manager.objects["test_object"], {"color": "red"})

    def test_get_object(self):
        manager = ObjectOrientedProgrammingManager()
        manager.create_object("test_object", {"color": "red"})
        obj = manager.get_object("test_object")
        self.assertIsNotNone(obj)
        self.assertEqual(obj, {"color": "red"})

    def test_update_object(self):
        manager = ObjectOrientedProgrammingManager()
        manager.create_object("test_object", {"color": "red"})
        manager.update_object("test_object", {"color": "blue"})
        obj = manager.get_object("test_object")
        self.assertIsNotNone(obj)
        self.assertEqual(obj, {"color": "blue"})

    def test_delete_object(self):
        manager = ObjectOrientedProgrammingManager()
        manager.create_object("test_object", {"color": "red"})
        manager.delete_object("test_object")
        self.assertNotIn("test_object", manager.objects)


if __name__ == "__main__":
    unittest.main()
