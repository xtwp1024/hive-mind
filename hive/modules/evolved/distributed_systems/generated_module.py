import random
import time
from typing import Any, Dict, List, Optional

class DistributedSystemsManager:
    """
    A Python module for managing distributed systems functionalities.
    """

    def __init__(self):
        self.systems_status: Dict[str, bool] = {}

    def add_system(self, system_id: str) -> None:
        """
        Adds a new system to the distributed system management.

        :param system_id: The unique identifier for the system.
        """
        if system_id in self.systems_status:
            raise ValueError(f"System with ID {system_id} already exists.")
        self.systems_status[system_id] = True

    def remove_system(self, system_id: str) -> None:
        """
        Removes a system from the distributed system management.

        :param system_id: The unique identifier for the system.
        """
        if system_id not in self.systems_status:
            raise ValueError(f"System with ID {system_id} does not exist.")
        del self.systems_status[system_id]

    def check_system_status(self, system_id: str) -> bool:
        """
        Checks the status of a system in the distributed system management.

        :param system_id: The unique identifier for the system.
        :return: True if the system is up, False otherwise.
        """
        return self.systems_status.get(system_id, False)

    def restart_system(self, system_id: str) -> None:
        """
        Restarts a system in the distributed system management.

        :param system_id: The unique identifier for the system.
        """
        if system_id not in self.systems_status:
            raise ValueError(f"System with ID {system_id} does not exist.")
        self.systems_status[system_id] = True

    def simulate_system_failure(self, system_id: str) -> None:
        """
        Simulates a system failure in the distributed system management.

        :param system_id: The unique identifier for the system.
        """
        if system_id not in self.systems_status:
            raise ValueError(f"System with ID {system_id} does not exist.")
        self.systems_status[system_id] = False

    def get_all_systems_status(self) -> Dict[str, bool]:
        """
        Returns the status of all systems in the distributed system management.

        :return: A dictionary with system IDs as keys and their status as values.
        """
        return self.systems_status

# Unit tests for the DistributedSystemsManager class
def test_add_system():
    manager = DistributedSystemsManager()
    manager.add_system("system1")
    assert "system1" in manager.systems_status

def test_remove_system():
    manager = DistributedSystemsManager()
    manager.add_system("system1")
    manager.remove_system("system1")
    assert "system1" not in manager.systems_status

def test_check_system_status():
    manager = DistributedSystemsManager()
    manager.add_system("system1")
    assert manager.check_system_status("system1") == True

def test_restart_system():
    manager = DistributedSystemsManager()
    manager.add_system("system1")
    manager.restart_system("system1")
    assert manager.check_system_status("system1") == True

def test_simulate_system_failure():
    manager = DistributedSystemsManager()
    manager.add_system("system1")
    manager.simulate_system_failure("system1")
    assert manager.check_system_status("system1") == False

def test_get_all_systems_status():
    manager = DistributedSystemsManager()
    manager.add_system("system1")
    manager.add_system("system2")
    status = manager.get_all_systems_status()
    assert "system1" in status and "system2" in status
    assert status["system1"] == True and status["system2"] == True

# Run the unit tests
if __name__ == "__main__":
    test_add_system()
    test_remove_system()
    test_check_system_status()
    test_restart_system()
    test_simulate_system_failure()
    test_get_all_systems_status()
    print("All unit tests passed.")