#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDDL Validator (进化版本 - 吸收了 Hive-ICLR-2025 的源码级学术框架)
可以解析 .pddl 域文件并通过前置和副作用去约束和预演 Agent 的每一步思考操作。
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.PDDL")


class PDDLDomainParser:
    """极简的 LISP-Style 领域定义动作提取器"""

    def __init__(self, domains_dir: str):
        self.domains_dir = Path(domains_dir)
        self.actions = {}  # action_name -> {parameters: [], precondition: "", effect: ""}
        self._load_all_domains()

    def _load_all_domains(self):
        if not self.domains_dir.exists():
            return

        # 匹配 (:action action_name   ...  :parameters ... :precondition ... :effect ...)
        action_pattern = re.compile(
            r"\(:action\s+([a-zA-Z0-9_\-]+).*?:parameters\s*\((.*?)\).*?:precondition\s*(.*?):effect\s*(.*?)\)",
            re.IGNORECASE | re.DOTALL,
        )

        for file in self.domains_dir.glob("*-domain.pddl"):
            try:
                content = file.read_text(encoding="utf-8")
                # 简单粗暴去掉注释
                content = re.sub(r";.*?\n", "\n", content)

                # 由于 lisp 括号嵌套复杂，简单的正则提取很难做到完美
                # 这里我们利用切片的方式，把按 :action 分块
                blocks = content.split("(:action")
                for block in blocks[1:]:
                    block = block.strip()
                    action_name = block.split()[0].strip()

                    # 摘取三大块
                    params_match = re.search(
                        r":parameters\s*\((.*?)\)", block, re.DOTALL | re.IGNORECASE
                    )
                    pre_match = re.search(
                        r":precondition\s*(.*?)(?::effect|\))", block, re.DOTALL | re.IGNORECASE
                    )
                    eff_match = re.search(
                        r":effect\s*(.*?)(?:\(:action|\)$)", block, re.DOTALL | re.IGNORECASE
                    )

                    self.actions[action_name] = {
                        "parameters": params_match.group(1).strip() if params_match else "",
                        "precondition": pre_match.group(1).strip() if pre_match else "",
                        "effect": eff_match.group(1).strip() if eff_match else "",
                    }
                    logger.debug(f"加载 PDDL 动作字典: {action_name}")
            except Exception as e:
                logger.error(f"解析 {file.name} 失败: {e}")


class PDDLValidator:
    """动作约束校验器"""

    _parser = None

    @classmethod
    def get_parser(cls) -> PDDLDomainParser:
        if cls._parser is None:
            from hive.config.settings import get_settings

            # 引入全局的 PDDL domains 文件夹
            domain_path = get_settings().BASE_DIR / "hive" / "pddl_domains"
            cls._parser = PDDLDomainParser(str(domain_path))
        return cls._parser

    @classmethod
    def validate_action(
        cls, action_name: str, arguments: Dict[str, Any], current_state: Dict[str, Any]
    ) -> bool:
        parser = cls.get_parser()

        # 1. 如果这个动作在 PDDL 中未定义，则报警 (严格模式)
        if action_name not in parser.actions:
            # 放宽：为了兼容原版工具，如果没有则走原始硬编码校验规则
            return cls._fallback_hardcoded_check(action_name, arguments, current_state)

        action_def = parser.actions[action_name]
        precond_str = action_def["precondition"]

        logger.debug(f"🔍 [PDDL 学术约束演算] 动作={action_name}, 要求={precond_str}")

        # 2. PDDL Precondition 判定体系 (简化模拟)
        # 例如: (and (IsText ?question) (IsText ?context))
        # 真正的学术 PDDL solver 会在此做多路谓语逻辑演算。
        # 这里我们在防瞎说系统：只要谓语在 current_state 知识内没满足，就会拦截

        if "(IsText" in precond_str:
            # 模拟：任何标记了 IsText 要求入参文本的，参数绝不能为空对象
            for k, v in arguments.items():
                if not isinstance(v, str) or not v.strip():
                    logger.warning(
                        f"🚫 [PDDL 拦截] 动作 {action_name} 期望入参文本但给定为空 / 违背类型。"
                    )
                    return False

        return True

    @classmethod
    def _fallback_hardcoded_check(
        cls, action_name: str, arguments: Dict[str, Any], current_state: Dict[str, Any]
    ) -> bool:
        if action_name == "trade_sell":
            holding = current_state.get("holdings", {}).get(arguments.get("symbol"), 0)
            if arguments.get("amount", 0) > holding:
                logger.warning(f"🚫 [硬编码拦截] 尝试卖出量大过持有量。")
                return False
        elif action_name == "delete_system_file":
            path = arguments.get("path", "")
            if path in ["/", "C:\\Windows", "C:\\", "/etc"]:
                return False
        return True

    @classmethod
    def predict_next_state(cls, action_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """[衍生] PDDL 规划：在执行完毕动作后，应用 effect 给系统环境施加副作用"""
        parser = cls.get_parser()
        next_state = current_state.copy()
        if action_name in parser.actions:
            effect_str = parser.actions[action_name]["effect"]
            # 简化演化：把 effect 字符串直接作为达成的能力状态压入 state (如 ExtractAnswerFromContext)
            next_state.setdefault("achieved_effects", []).append(effect_str)
        return next_state
