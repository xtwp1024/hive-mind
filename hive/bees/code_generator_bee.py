#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeGeneratorBee - 知识驱动的代码生成器
基于 ChromaDB 知识库和 GLM-4 大脑生成代码
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from hive.bees.base_bee import BaseBee
from hive.core.vector_vault import get_vector_vault
from hive.utils.brain_interface import BrainInterface

logger = logging.getLogger("Hive.CodeGenerator")

class CodeGeneratorBee(BaseBee):
    """代码生成蜂 - 基于知识库的智能代码生成"""
    
    def __init__(self, dna, config=None, result_queue=None):
        super().__init__(dna, config, result_queue)
        self.vault = self._connect_vault()
        self.brain = self._connect_brain()
    
    def _connect_vault(self):
        try:
            v = get_vector_vault()
            logger.info(f"📚 知识库已连接: {v.get_stats().get('total_documents', 0)} 条记忆")
            return v
        except Exception as e:
            logger.error(f"❌ 知识库连接失败: {e}")
            return None

    def _connect_brain(self):
        try:
            b = BrainInterface()
            logger.info("🧠 大脑已连接")
            return b
        except Exception as e:
            logger.error(f"❌ 大脑连接失败: {e}")
            return None

    def specialized_task(self, target_path: Path) -> Dict[str, Any]:
        logger.info("⚠️ CodeGeneratorBee 应通过 generate_code() 方法使用")
        return {"status": "use_generate_code_method"}

    def generate_code(self, requirement: str, output_dir: str = "./generated", language: str = "python", max_ref: int = 10) -> Dict:
        logger.info(f"🎯 开始生成代码: {requirement}")
        refs = self._retrieve_knowledge(requirement, max_ref)
        code_data = self._generate_with_brain(requirement, refs, language)
        
        if not code_data: return {"status": "FAILED", "error": "代码生成失败"}
        
        saved = self._save_code(code_data, Path(output_dir))
        return {
            "status": "SUCCESS", "requirement": requirement,
            "references_used": len(refs), "files_generated": saved,
            "output_dir": output_dir
        }

    def _retrieve_knowledge(self, query: str, limit: int) -> List[Dict]:
        if not self.vault: return []
        try:
            results = self.vault.search(query, limit=limit)
            return [{"content": r.get("document", "")[:500], "similarity": 1-r.get("distance", 1.0)} for r in results]
        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            return []

    def _generate_with_brain(self, req: str, refs: List[Dict], lang: str) -> Optional[Dict]:
        if not self.brain: return None
        
        sys_prompt = self._construct_system_prompt(lang)
        user_prompt = self._construct_user_prompt(req, refs, lang)
        
        try:
            resp = self.brain.consult(user_prompt, sys_prompt)
            return self._parse_brain_response(resp) if resp else None
        except Exception as e:
            logger.error(f"❌ 生成异常: {e}")
            return None

    def _construct_system_prompt(self, lang: str) -> str:
        return f"""你是一个专业的 {lang} 代码生成专家。
输出格式（JSON）:
{{
  "files": {{ "filename": "code content" }},
  "explanation": "说明"
}}"""

    def _construct_user_prompt(self, req: str, refs: List[Dict], lang: str) -> str:
        ref_text = "\n".join([f"Ref {i+1}: {r['content'][:300]}..." for i, r in enumerate(refs[:5])])
        return f"用户需求: {req}\n语言: {lang}\n参考:\n{ref_text}\n请生成 JSON。"

    def _parse_brain_response(self, response: str) -> Dict:
        try:
            clean_resp = self._extract_json_block(response)
            data = json.loads(clean_resp)
            return data if "files" in data else {"files": {"generated.py": clean_resp}, "explanation": "Wrapped"}
        except json.JSONDecodeError:
            return {"files": {"generated.txt": response}, "explanation": "Raw response"}

    def _extract_json_block(self, text: str) -> str:
        if "```json" in text: return text.split("```json")[1].split("```")[0].strip()
        if "```" in text: return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def _save_code(self, data: Dict, out_dir: Path) -> List[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, content in data.get("files", {}).items():
            try:
                # 防止路径穿越：确保解析后的路径仍在 out_dir 内
                safe_path = (out_dir / name).resolve()
                if not str(safe_path).startswith(str(out_dir.resolve())):
                    logger.warning(f"拒绝路径穿越文件名: {name}")
                    continue
                safe_path.parent.mkdir(parents=True, exist_ok=True)
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(content)
                saved.append(str(safe_path))
            except Exception as e:
                logger.error(f"保存失败 {name}: {e}")
        return saved

def generate_code(requirement: str, output_dir: str = "./generated") -> Dict:
    from hive.core.dna import BeeDNA
    dna = BeeDNA(species="bee", role="code_generator", id="CodeGen_001", target_name="code_generation")
    return CodeGeneratorBee(dna).generate_code(requirement, output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(generate_code("Create a FastAPI server"), indent=2))
