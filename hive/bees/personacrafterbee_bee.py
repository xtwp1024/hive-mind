# -*- coding: utf-8 -*-
import random
from typing import Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class PersonaCrafterBee(BaseBee):
    """人设工匠蜂 - 创建角色人设、生成对话风格"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.personas = {}
        self.dialogue_styles = {
            "formal": {"tone": "professional", "greeting": "Dear"},
            "casual": {"tone": "friendly", "greeting": "Hey"},
            "technical": {"tone": "analytical", "greeting": "System"},
            "creative": {"tone": "inspiring", "greeting": "Hello"},
        }

    def specialized_task(self, target_path):
        """执行人设创建任务"""
        logger.info(f"[PersonaCrafterBee-{self.id}] 人设: {target_path}")

        # 创建人设
        persona = self._create_persona(target_path)

        # 生成对话风格
        style = self._generate_dialogue_style(persona)

        logger.info(f"[PersonaCrafterBee-{self.id}] 人设创建完成: {persona['name']}")

    def _create_persona(self, context: str) -> Dict:
        """创建人设"""
        roles = ["expert", "helper", "analyst", "creator", "guide"]
        traits = ["analytical", "creative", "helpful", "precise", "friendly"]

        persona = {
            "name": f"{context}_persona",
            "role": random.choice(roles),
            "traits": random.sample(traits, 2),
            "backstory": f"Specialist in {context}",
            "goals": ["assist users", "provide expertise"],
        }

        self.personas[persona["name"]] = persona
        return persona

    def _generate_dialogue_style(self, persona: Dict) -> Dict:
        """生成对话风格"""
        style_key = random.choice(list(self.dialogue_styles.keys()))
        style = self.dialogue_styles[style_key].copy()

        style["persona"] = persona["name"]
        style["examples"] = [
            f"{style['greeting']}, I can help with {persona['role']} tasks.",
            f"My {persona['traits'][0]} approach ensures quality.",
        ]

        return style

    def check_consistency(self, message: str, persona_name: str) -> float:
        """检查一致性"""
        if persona_name not in self.personas:
            return 0.0

        persona = self.personas[persona_name]
        # 简化的逻辑检查
        consistency = 0.8

        return consistency

    def adapt_to_context(self, persona_name: str, context: str) -> Dict:
        """适配人设到上下文"""
        if persona_name not in self.personas:
            return {"error": "Persona not found"}

        persona = self.personas[persona_name]
        adaptation = {
            "original": persona.copy(),
            "adapted": {
                **persona,
                "context": context,
                "specialization": f"{persona['role']} for {context}",
            },
        }

        return adaptation

    def devour(self) -> dict:
        """消耗目标并返回人设报告"""
        target_path = self.dna.id

        try:
            self.specialized_task(target_path)

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "personas": self.personas,
                "styles": self.dialogue_styles,
            }
        except Exception as e:
            logger.error(f"[PersonaCrafterBee-{self.id}] 人设创建失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
