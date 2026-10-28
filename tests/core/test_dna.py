from hive.core.dna import BeeDNA, DNA
import pytest

def test_dna_initialization():
    dna = BeeDNA(species="TestBee", role="worker", id="test_001")
    assert dna.species == "TestBee"
    assert dna.role == "worker"
    assert dna.fitness == 0.0

def test_dna_mutation():
    dna = BeeDNA(species="TestBee", role="worker", id="test_001", learning_rate=0.5)
    
    mutated_dna = dna.mutate(mutation_rate=1.0)
    
    assert mutated_dna.id != dna.id
    assert mutated_dna.generation == dna.generation + 1
    assert 0.0 <= mutated_dna.learning_rate <= 1.0

def test_dna_crossover():
    dna1 = BeeDNA(species="TestBee", id="1", learning_rate=0.1)
    dna2 = BeeDNA(species="TestBee", id="2", learning_rate=0.9)
    
    child = dna1.crossover(dna2)
    assert child.generation == max(dna1.generation, dna2.generation) + 1
    assert child.learning_rate == pytest.approx(0.5)

def test_dna_initialization_with_all_params():
    dna = BeeDNA(
        species="MinerBee",
        role="miner",
        id="miner_001",
        target_name="TestProject",
        target_url="https://github.com/test/repo",
        learning_rate=0.7,
        mutation_gene="python",
        fitness=0.85
    )
    assert dna.species == "MinerBee"
    assert dna.role == "miner"
    assert dna.target_name == "TestProject"
    assert dna.learning_rate == 0.7
    assert dna.fitness == 0.85

def test_dna_validation():
    dna = BeeDNA(species="ValidBee", role="miner", id="valid_001", 
                  mining_capability=0.8, learning_rate=0.5)
    assert dna.validate() == True

def test_dna_validation_invalid_capability():
    dna = BeeDNA(species="InvalidBee", role="miner", id="invalid_001", 
                  mining_capability=1.5)
    assert dna.validate() == False

def test_dna_mutation_rate_zero():
    dna = BeeDNA(species="TestBee", role="worker", id="zero_mut", learning_rate=0.5)
    original_lr = dna.learning_rate
    mutated = dna.mutate(mutation_rate=0.0)
    
    assert mutated.learning_rate == original_lr
    assert mutated.generation == dna.generation + 1

def test_base_dna_initialization():
    dna = DNA(species="BaseBee", generation=1, fitness=0.5)
    assert dna.species == "BaseBee"
    assert dna.generation == 1
    assert dna.fitness == 0.5

def test_base_dna_validate():
    dna = DNA(species="TestDNA", mining_capability=0.5, learning_rate=0.5)
    assert dna.validate() == True

def test_dna_crossover_different_species():
    dna1 = BeeDNA(species="BeeA", id="1")
    dna2 = BeeDNA(species="BeeB", id="2")
    
    child = dna1.crossover(dna2)
    assert child.species in ["BeeA", "BeeB"]

def test_dna_default_values():
    dna = BeeDNA(species="DefaultBee", id="default")
    
    assert dna.role == "worker"
    assert dna.target_name == ""
    assert dna.target_url == ""
    assert dna.learning_rate == 0.5

def test_dna_multiple_mutations():
    dna = BeeDNA(species="MultiMut", id="multi", learning_rate=0.5)
    
    mutated1 = dna.mutate(mutation_rate=0.5)
    mutated2 = mutated1.mutate(mutation_rate=0.5)
    
    assert mutated2.generation == dna.generation + 2

def test_dna_capabilities():
    dna = BeeDNA(species="CapBee", id="cap", mining_capability=0.8)
    assert dna.mining_capability == 0.8

def test_dna_metadata():
    dna = BeeDNA(species="MetaBee", id="meta", metadata={"key": "value"})
    assert dna.metadata["key"] == "value"
