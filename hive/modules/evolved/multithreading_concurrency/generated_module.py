import threading
from typing import List
import unittest

class MultithreadingConcurrencyManager:
    """
    MultithreadingConcurrencyManager is a class that provides methods to demonstrate multithreading concurrency in Python.
    It includes methods to run tasks concurrently, check the status of threads, and join threads to the main thread.
    """
    
    def __init__(self):
        self.threads = []
    
    def run_concurrently(self, tasks: List[callable], *args, **kwargs) -> None:
        """
        Run multiple tasks concurrently using separate threads.
        
        :param tasks: List of callable functions to be executed concurrently.
        :param args: Variable length argument list to be passed to the tasks.
        :param kwargs: Arbitrary keyword arguments to be passed to the tasks.
        """
        for task in tasks:
            thread = threading.Thread(target=task, args=args, kwargs=kwargs)
            self.threads.append(thread)
            thread.start()
    
    def join_threads(self) -> None:
        """
        Wait for all threads to complete.
        """
        for thread in self.threads:
            thread.join()
    
    def get_thread_status(self) -> List[str]:
        """
        Get the status of all threads.
        
        :return: List of status strings for each thread.
        """
        statuses = []
        for thread in self.threads:
            if thread.is_alive():
                statuses.append("Running")
            else:
                statuses.append("Completed")
        return statuses

class TestMultithreadingConcurrencyManager(unittest.TestCase):
    """
    Unit tests for the MultithreadingConcurrencyManager class.
    """
    
    def test_run_concurrently(self):
        manager = MultithreadingConcurrencyManager()
        tasks = [self.thread_task, self.thread_task]
        manager.run_concurrently(tasks)
        manager.join_threads()
        self.assertEqual(manager.get_thread_status(), ["Completed", "Completed"])
    
    def test_join_threads(self):
        manager = MultithreadingConcurrencyManager()
        tasks = [self.thread_task, self.thread_task]
        manager.run_concurrently(tasks)
        manager.join_threads()
        self.assertEqual(manager.get_thread_status(), ["Completed", "Completed"])
    
    def test_get_thread_status(self):
        manager = MultithreadingConcurrencyManager()
        tasks = [self.thread_task, self.thread_task]
        manager.run_concurrently(tasks)
        statuses = manager.get_thread_status()
        self.assertTrue(all(status == "Completed" for status in statuses))
    
    @staticmethod
    def thread_task():
        # Simulate a long-running task
        import time
        time.sleep(1)

if __name__ == "__main__":
    unittest.main()