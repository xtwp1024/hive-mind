# -*- coding: utf-8 -*-
import random
import time
from pathlib import Path

from hive.bees.base_bee import BaseBee


class RalphBee(BaseBee):
    """
    RalphBee: A chaotic, serendipitous agent resembling Ralph Wiggum.
    Scans for simple patterns ("worms") while providing questionable wisdom.
    """

    QUOTES = [
        "I'm helping!",
        "My cat's breath smells like cat food.",
        "Me fail English? That's unpossible!",
        "I bent my wookie.",
        "It tastes like burning!",
        "I'm a unit tester!",
        "Look in the tunk.",
        "I found a moon rock in my nose.",
        "The doctor said I wouldn't have so many nose bleeds if I kept my finger outta there.",
        "I chose the sticky one!",
        "I'm learning!",
        "Hi, Super Nintendo Chalmers!",
    ]

    def specialized_task(self, target_path: Path) -> dict:
        """
        Ralph's specific task: Find 'worms' (TODOs) and 'burning' things (errors).
        """
        quote = random.choice(self.QUOTES)
        self.log(f"🧠 [Ralph] Thinking: '{quote}'")

        worms_found = []
        burning_sensations = []

        # Simulate Ralph getting distracted
        time.sleep(random.uniform(0.5, 1.5))

        if not target_path.exists():
            return {"ralph_says": "Where did the building go?", "quote": quote}

        # Naive scan for files
        py_files = list(target_path.rglob("*.py"))

        if not py_files:
            return {"ralph_says": "I eated the purple berries.", "quote": quote}

        # Pick random files to "taste" (limit to 5 to avoid brain freeze)
        files_to_taste = random.sample(py_files, min(len(py_files), 5))

        for file_path in files_to_taste:
            try:
                self.log(f"👅 [Ralph] Tasting {file_path.name}...")
                content = file_path.read_text(encoding="utf-8", errors="ignore")

                if "pass" in content:
                    worms_found.append(f"{file_path.name} has a placeholder worm!")

                if "TODO" in content or "todo" in content:
                    worms_found.append(f"{file_path.name} needs to do things!")

                if "FIXME" in content:
                    burning_sensations.append(f"{file_path.name} is broken!")

            except Exception as e:
                self.log(f"🔥 [Ralph] It tastes like burning! ({e})")
                burning_sensations.append(str(e))

        return {
            "status": "SUCCESS",
            "ralph_says": quote,
            "worms_found": worms_found,
            "burning_things": burning_sensations,
            "tasted_files": [f.name for f in files_to_taste],
        }

    def log(self, msg: str):
        # Override log to add Ralph flair if needed, using standard logger for now
        # Access the module level logger or print? BaseBee uses logger.
        # We'll use the inherited logger via module import or assume BaseBee sets it up.
        # BaseBee uses module level logger, so we can define one here or use base prints.
        # Actually BaseBee uses `logger.info`, which refers to the module's logger.
        # Let's import get_logger.
        from hive.utils.logger import get_logger

        l = get_logger("RalphBee")
        l.info(msg)
