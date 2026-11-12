import random
import time
from typing import Any, Dict, List


class DistributedSystemsManager:
    """A class for managing distributed systems with core functionalities."""

    def __init__(self):
        """Initialize the DistributedSystemsManager."""
        self.nodes = {}
        self.failures = []

    def add_node(self, node_id: str, status: str = "active"):
        """
        Add a new node to the distributed system.

        :param node_id: Unique identifier for the node
        :param status: Status of the node ('active' or 'inactive')
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with ID {node_id} already exists.")
        self.nodes[node_id] = {"status": status, "load": 0}

    def remove_node(self, node_id: str):
        """
        Remove a node from the distributed system.

        :param node_id: Unique identifier for the node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist.")
        del self.nodes[node_id]

    def update_node_status(self, node_id: str, status: str):
        """
        Update the status of a node.

        :param node_id: Unique identifier for the node
        :param status: New status of the node ('active' or 'inactive')
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist.")
        self.nodes[node_id]["status"] = status

    def add_load(self, node_id: str, load: int):
        """
        Add load to a node.

        :param node_id: Unique identifier for the node
        :param load: Amount of load to add
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist.")
        self.nodes[node_id]["load"] += load

    def distribute_load(self):
        """
        Distribute load across all active nodes.
        """
        for node_id, node_info in self.nodes.items():
            if node_info["status"] == "active":
                self.add_load(node_id, random.randint(1, 10))

    def handle_failures(self):
        """
        Handle failures by marking nodes as inactive.
        """
        for node_id in self.failures:
            if node_id in self.nodes:
                self.nodes[node_id]["status"] = "inactive"
        self.failures = []

    def get_node_status(self, node_id: str) -> str:
        """
        Get the status of a node.

        :param node_id: Unique identifier for the node
        :return: Status of the node ('active' or 'inactive')
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist.")
        return self.nodes[node_id]["status"]

    def simulate_failure(self, node_id: str):
        """
        Simulate a node failure.

        :param node_id: Unique identifier for the node
        """
        self.failures.append(node_id)


# Unit tests for the module
import unittest


class TestDistributedSystemsManager(unittest.TestCase):

    def setUp(self):
        self.manager = DistributedSystemsManager()
        self.manager.add_node("node1")
        self.manager.add_node("node2")

    def test_add_node(self):
        self.manager.add_node("node3")
        self.assertIn("node3", self.manager.nodes)

    def test_remove_node(self):
        self.manager.remove_node("node2")
        self.assertNotIn("node2", self.manager.nodes)

    def test_update_node_status(self):
        self.manager.update_node_status("node1", "inactive")
        self.assertEqual(self.manager.get_node_status("node1"), "inactive")

    def test_add_load(self):
        self.manager.add_load("node1", 5)
        self.assertEqual(self.manager.nodes["node1"]["load"], 5)

    def test_distribute_load(self):
        self.manager.distribute_load()
        for node_id, node_info in self.manager.nodes.items():
            self.assertGreaterEqual(node_info["load"], 0)

    def test_get_node_status(self):
        self.manager.update_node_status("node1", "inactive")
        self.assertEqual(self.manager.get_node_status("node1"), "inactive")

    def test_simulate_failure(self):
        self.manager.simulate_failure("node1")
        self.assertIn("node1", self.manager.failures)


if __name__ == "__main__":
    unittest.main()
