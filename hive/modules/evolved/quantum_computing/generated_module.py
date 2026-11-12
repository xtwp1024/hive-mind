from typing import Any, List

import qiskit


class QuantumComputingManager:
    """
    QuantumComputingManager is a class that provides an interface for quantum computing operations.
    It uses the Qiskit library to create and manipulate quantum circuits.
    """

    def __init__(self):
        """
        Initializes the QuantumComputingManager with a default quantum circuit.
        """
        self.circuit = qiskit.circuit.Circuit()

    def add_qubit(self, index: int) -> None:
        """
        Adds a qubit to the quantum circuit at the specified index.

        :param index: The index at which to add the qubit.
        """
        self.circuit.add_qubit(index)

    def add_classical_bit(self, index: int) -> None:
        """
        Adds a classical bit to the quantum circuit at the specified index.

        :param index: The index at which to add the classical bit.
        """
        self.circuit.add_clbit(index)

    def apply_gate(self, gate: str, qubits: List[int], params: List[Any] = None) -> None:
        """
        Applies a quantum gate to the specified qubits in the circuit.

        :param gate: The name of the quantum gate to apply.
        :param qubits: The list of qubits to apply the gate to.
        :param params: The parameters for the gate, if any.
        """
        if params:
            self.circuit.apply_gate(gate, qubits, params)
        else:
            self.circuit.apply_gate(gate, qubits)

    def measure(self, qubits: List[int], outcomes: List[int]) -> None:
        """
        Measures the specified qubits and assigns the outcomes to the corresponding classical bits.

        :param qubits: The list of qubits to measure.
        :param outcomes: The list of outcomes to assign to the measured qubits.
        """
        for qubit, outcome in zip(qubits, outcomes):
            self.circuit.measure(qubit, outcome)

    def execute(self, backend: qiskit.providers.backends.BaseBackend) -> qiskit.result.Result:
        """
        Executes the quantum circuit on the specified backend and returns the results.

        :param backend: The quantum backend to execute the circuit on.
        :return: The results of the quantum circuit execution.
        """
        return qiskit.execute(self.circuit, backend)


# Unit tests for QuantumComputingManager
def test_add_qubit():
    manager = QuantumComputingManager()
    manager.add_qubit(0)
    assert len(manager.circuit.qubits) == 1, "Failed to add qubit."


def test_add_classical_bit():
    manager = QuantumComputingManager()
    manager.add_classical_bit(0)
    assert len(manager.circuit.clbits) == 1, "Failed to add classical bit."


def test_apply_gate():
    manager = QuantumComputingManager()
    manager.add_qubit(0)
    manager.apply_gate("h", [0])
    assert "h" in manager.circuit.data, "Failed to apply gate."


def test_measure():
    manager = QuantumComputingManager()
    manager.add_qubit(0)
    manager.add_classical_bit(0)
    manager.measure([0], [0])
    assert "measure" in manager.circuit.data, "Failed to measure qubit."


def test_execute():
    manager = QuantumComputingManager()
    manager.add_qubit(0)
    manager.apply_gate("h", [0])
    backend = qiskit.Aer.get_backend("qasm_simulator")
    result = manager.execute(backend)
    assert isinstance(result, qiskit.result.Result), "Failed to execute circuit."


# Run unit tests
if __name__ == "__main__":
    test_add_qubit()
    test_add_classical_bit()
    test_apply_gate()
    test_measure()
    test_execute()
    print("All tests passed.")
