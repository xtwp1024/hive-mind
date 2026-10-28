import sys
import multiprocessing
from pathlib import Path
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from hive.core.queen import QueenBee
from hive.utils.logger import setup_logging

def test_v2_integration():
    setup_logging(log_level="DEBUG")
    logger = logging.getLogger("Test")
    
    logger.info("Initializing Queen...")
    q = multiprocessing.Queue()
    queen = QueenBee(q)
    
    logger.info("Checking V2 Dispatcher Actors...")
    actors = queen.dispatcher.actors
    logger.info(f"Found {len(actors)} registered actors.")
    
    for actor in actors:
        logger.info(f"- Actor: {actor.name} (Role: {actor.role})")
        logger.info(f"  Capabilities: {actor.capabilities()}")
        
    if len(actors) > 0:
        print("✅ SUCCESS: Queen successfully registered V1 bees as V2 actors.")
    else:
        print("❌ FAILURE: No actors registered.")

if __name__ == "__main__":
    test_v2_integration()
