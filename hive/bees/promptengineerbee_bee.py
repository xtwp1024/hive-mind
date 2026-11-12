# -*- coding: utf-8 -*-
import hashlib
from typing import Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class PromptEngineerBee(BaseBee):
    """提示工程师蜂 - 优化提示词、测试提示效果"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.prompt_templates = {}
        self.prompt_history = []

    def specialized_task(self, target_path):
        """执行提示工程任务"""
        logger.info(f"[PromptEngineerBee-{self.id}] 提示: {target_path}")

        # 优化提示词
        optimized = self._optimize_prompt(target_path)

        # 保存模板
        self._save_template(target_path, optimized)

        logger.info(f"[PromptEngineerBee-{self.id}] 提示优化完成")

    def _optimize_prompt(self, original: str) -> Dict:
        """优化提示词"""
        # 添加上下文
        context_enhanced = (
            f"Context: {original}\n\nTask: Optimize and improve this prompt for better results."
        )

        # 添加约束
        constraints = "\n\nConstraints: Be precise, actionable, and clear."

        # 添加格式
        format_instruction = "\n\nFormat: Provide structured output."

        optimized = context_enhanced + constraints + format_instruction

        return {
            "original": original,
            "optimized": optimized,
            "improvements": ["Added context", "Added constraints", "Added format instructions"],
        }

    def _save_template(self, name: str, prompt_data: Dict):
        """保存模板"""
        template_id = f"template_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        self.prompt_templates[template_id] = {
            "name": name,
            "prompt": prompt_data["optimized"],
            "version": "1.0",
            "created_at": "prompt_timestamp",
        }

        self.prompt_history.append({"template_id": template_id, "action": "created"})

    def test_prompt(self, prompt: str) -> Dict:
        """测试提示效果"""
        # 模拟测试
        result = {
            "prompt": prompt,
            "quality_score": 0.85,
            "clarity": 0.90,
            "specificity": 0.80,
            "estimated_success_rate": 0.88,
        }

        return result

    def manage_templates(self) -> List[Dict]:
        """管理模板"""
        return list(self.prompt_templates.values())

    def get_template(self, template_id: str) -> Dict:
        """获取模板"""
        return self.prompt_templates.get(template_id, {})

    def devour(self) -> dict:
        """消耗目标并返回提示工程报告"""
        target_path = self.dna.id

        try:
            self.specialized_task(target_path)

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "templates": len(self.prompt_templates),
                "history": len(self.prompt_history),
            }
        except Exception as e:
            logger.error(f"[PromptEngineerBee-{self.id}] 提示工程失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
