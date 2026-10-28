class Block:
    def __init__(self, index, transactions, timestamp, previous_hash):
        """Initialize a new block."""
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Calculate the hash of the block."""
        block_string = f"{self.index}{self.transactions}{self.timestamp}{self.previous_hash}"
        return hash(block_string)

class Blockchain:
    def __init__(self):
        """Initialize a new blockchain."""
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create a new genesis block for the blockchain."""
        genesis_block = Block(0, [], 0, "0")
        self.chain.append(genesis_block)

    def add_block(self, transactions):
        """Add a new block to the blockchain."""
        previous_block = self.chain[-1]
        new_block = Block(index=len(self.chain), transactions=transactions, timestamp=0, previous_hash=previous_block.hash)
        self.chain.append(new_block)

    def is_chain_valid(self):
        """Check if the blockchain is valid."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

class BlockchainDevelopmentManager:
    """Manager class for blockchain development."""

    def __init__(self):
        """Initialize a new BlockchainDevelopmentManager instance."""
        self.blockchain = Blockchain()

    def add_transaction(self, transaction):
        """Add a new transaction to the blockchain."""
        self.blockchain.add_block([transaction])

    def validate_blockchain(self):
        """Validate the blockchain."""
        return self.blockchain.is_chain_valid()

    def get_chain(self):
        """Get the current blockchain."""
        return self.blockchain.chain

# Unit tests
def test_blockchain_creation():
    manager = BlockchainDevelopmentManager()
    assert isinstance(manager.blockchain, Blockchain)
    assert manager.blockchain.chain == [manager.blockchain.chain[0]]

def test_blockchain_add_transaction():
    manager = BlockchainDevelopmentManager()
    manager.add_transaction('Transaction1')
    assert len(manager.blockchain.chain) == 2
    assert manager.blockchain.chain[1].transactions == ['Transaction1']

def test_blockchain_validation():
    manager = BlockchainDevelopmentManager()
    manager.add_transaction('Transaction1')
    assert manager.validate_blockchain()
    manager.blockchain.chain[1].transactions[0] = 'InvalidTransaction'
    assert not manager.validate_blockchain()

# Run unit tests
test_blockchain_creation()
test_blockchain_add_transaction()
test_blockchain_validation()