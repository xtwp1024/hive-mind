class Node:
    """Represents a node in a linked list."""

    def __init__(self, data: int):
        """Initialize a new node with given data."""
        self.data = data
        self.next = None


class LinkedList:
    """Represents a singly linked list."""

    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None

    def append(self, data: int) -> None:
        """Append a new node with the given data to the end of the list."""
        if not self.head:
            self.head = Node(data)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)

    def find(self, key: int) -> Node:
        """Find a node with the given key in the list."""
        current = self.head
        while current:
            if current.data == key:
                return current
            current = current.next
        raise ValueError(f"Node with data {key} not found in the list.")

    def remove(self, key: int) -> None:
        """Remove a node with the given key from the list."""
        current = self.head
        previous = None
        while current and current.data != key:
            previous = current
            current = current.next
        if previous is None:
            self.head = current.next
        elif current:
            previous.next = current.next

    def __str__(self) -> str:
        """Return a string representation of the linked list."""
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements)


def bubble_sort(arr: list) -> list:
    """Sort a list using the bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick_sort(arr: list) -> list:
    """Sort a list using the quick sort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: list) -> list:
    """Sort a list using the merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left: list, right: list) -> list:
    """Merge two sorted lists into one sorted list."""
    merged, left_idx, right_idx = [], 0, 0
    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] < right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1
    merged.extend(left[left_idx:])
    merged.extend(right[right_idx:])
    return merged


# Unit tests
if __name__ == "__main__":
    # Test LinkedList
    linked_list = LinkedList()
    linked_list.append(3)
    linked_list.append(1)
    linked_list.append(4)
    linked_list.append(1)
    assert str(linked_list) == "3 -> 1 -> 4 -> 1", "LinkedList test failed"

    node = linked_list.find(3)
    assert node.data == 3, "LinkedList find test failed"

    linked_list.remove(1)
    assert str(linked_list) == "3 -> 4 -> 1", "LinkedList remove test failed"

    # Test bubble_sort
    assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [
        11,
        12,
        22,
        25,
        34,
        64,
        90,
    ], "Bubble sort test failed"

    # Test quick_sort
    assert quick_sort([64, 34, 25, 12, 22, 11, 90]) == [
        11,
        12,
        22,
        25,
        34,
        64,
        90,
    ], "Quick sort test failed"

    # Test merge_sort
    assert merge_sort([64, 34, 25, 12, 22, 11, 90]) == [
        11,
        12,
        22,
        25,
        34,
        64,
        90,
    ], "Merge sort test failed"

    print("All tests passed.")
