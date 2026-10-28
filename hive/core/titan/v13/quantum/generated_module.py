import random
from typing import Iterable, Any

class Agent:
    """
    Agent class represents an individual in the swarm, capable of processing sentiment data.
    """
    def __init__(self, initial_sentiment: float):
        self.sentiment = initial_sentiment

    def update_sentiment(self, new_sentiment: float, noise_level: float = 0.1) -> None:
        """
        Update the agent's sentiment based on new data and add noise to simulate real-world randomness.
        """
        self.sentiment = new_sentiment + random.uniform(-noise_level, noise_level)

    def get_sentiment(self) -> float:
        """
        Get the current sentiment of the agent.
        """
        return self.sentiment


class Swarm:
    """
    Swarm class represents the collective behavior of agents, capable of producing a consensus sentiment.
    """
    def __init__(self, agents: list):
        self.agents = agents

    def calculate_consensus(self) -> float:
        """
        Calculate the consensus sentiment of the swarm.
        """
        total_sentiment = sum(agent.get_sentiment() for agent in self.agents)
        return total_sentiment / len(self.agents)


class SwarmIntelligence:
    """
    SwarmIntelligence class provides methods to simulate the swarm's behavior and perform sentiment analysis.
    """
    def __init__(self, sentiment_data: Iterable[float]):
        self.agents = [Agent(sentiment) for sentiment in sentiment_data]

    def run_simulation(self, iterations: int = 100) -> float:
        """
        Run the simulation for a specified number of iterations to allow agents to update their sentiments.
        """
        for _ in range(iterations):
            for agent in self.agents:
                # Simulate real-world noise in sentiment data
                new_sentiment = random.uniform(-0.1, 0.1)
                agent.update_sentiment(new_sentiment)
        
        return Swarm(self.agents).calculate_consensus()

# Example usage
if __name__ == "__main__":
    sentiment_data = [0.5, 0.6, -0.2, 0.7, -0.3]
    swarm_intelligence = SwarmIntelligence(sentiment_data)
    consensus_sentiment = swarm_intelligence.run_simulation()
    print(f"The consensus sentiment is: {consensus_sentiment}")