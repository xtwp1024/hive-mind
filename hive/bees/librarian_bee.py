#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图书管理员蜂- LibrarianBee
总结和组织项目知识，提取关键信息
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.config.settings import HiveSettings, get_settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class LibrarianBee(BaseBee):
    """图书管理员蜂 - 负责总结和组织项目知识"""

    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        self.settings = get_settings()
        self.base_dir = Path(self.settings.BASE_DIR)
        self.summaries = []
        self.knowledge_index = {}

    def specialized_task(self, target_path: Path):
        logger.info(f"📚 开始整理项目: {self.dna.target_name}")
        try:
            doc_files = self._find_document_files(target_path)
            if not doc_files:
                logger.warning(f"⚠️ 未找到文档文件: {self.dna.target_name}")
                return

            summary = self._create_project_summary(target_path, doc_files)
            self._process_summarization(target_path, summary)

            logger.info(f"✅ 项目整理完成: {self.dna.target_name}")
            logger.info(f"   文档数: {len(doc_files)}, 总大小: {summary['total_size']}")
        except Exception as e:
            logger.error(f"❌ 图书管理任务失败: {e}")

    def _create_project_summary(self, target_path: Path, doc_files: List[Path]) -> Dict:
        project_summary = {
            "name": self.dna.target_name,
            "path": str(target_path),
            "documents": [],
            "total_size": 0,
            "tech_stack": self._analyze_tech_stack(target_path),
        }

        for doc_file in doc_files:
            info = self._process_single_doc(doc_file)
            if info:
                project_summary["documents"].append(info)
                project_summary["total_size"] += info["size"]
        return project_summary

    def _process_single_doc(self, doc_file: Path) -> Dict:
        try:
            with open(doc_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {
                "filename": doc_file.name,
                "size": doc_file.stat().st_size,
                "summary": self._generate_summary(content),
            }
        except Exception as e:
            logger.error(f"读取文档失败 {doc_file.name}: {e}")
            return None

    def _process_summarization(self, target_path: Path, summary: Dict):
        self._save_project_summary(target_path, summary)
        self._build_knowledge_index(target_path, summary)

    def _find_document_files(self, target_path: Path) -> list:
        found_docs = []
        # Search Root
        for ext in ["README.md", "LICENSE", "CONTRIBUTING.md", "CHANGELOG.md"]:
            if (target_path / ext).exists():
                found_docs.append(target_path / ext)
            if (target_path / ext.lower()).exists():
                found_docs.append(target_path / ext.lower())

        # Search Subdirs
        for d in ["docs", "documentation"]:
            if (target_path / d).exists():
                found_docs.extend((target_path / d).glob("*.md"))

        return found_docs

    def _analyze_tech_stack(self, target_path: Path) -> Dict[str, str]:
        stack = {"language": "Unknown", "frameworks": [], "tools": [], "dependencies": []}
        try:
            self._analyze_python_stack(target_path, stack)
            self._analyze_node_stack(target_path, stack)
        except Exception as e:
            logger.debug(f"技术栈分析失败: {e}")
        return stack

    def _analyze_python_stack(self, target_path: Path, stack: Dict):
        if list(target_path.glob("*.py")):
            stack["language"] = "Python"
            rf = target_path / "requirements.txt"
            if rf.exists():
                with open(rf, "r", errors="ignore") as f:
                    stack["dependencies"] = [
                        line.strip() for line in f if line.strip() and not line.startswith("#")
                    ]

            for f in ["config.yaml", "settings.py"]:
                if (target_path / f).exists():
                    stack["frameworks"].append(f"Config: {f}")

    def _analyze_node_stack(self, target_path: Path, stack: Dict):
        pj = target_path / "package.json"
        if pj.exists():
            with open(pj, "r", errors="ignore") as f:
                data = json.load(f)
                stack["language"] = data.get("name", "Node.js")
                if "dependencies" in data:
                    stack["dependencies"].extend(data["dependencies"].keys())

    def _generate_summary(self, content: str) -> str:
        return content[:500] + "..." if len(content) > 500 else content

    def _save_project_summary(self, target_path: Path, summary: Dict):
        try:
            sp = target_path / "PROJECT_SUMMARY.md"
            with open(sp, "w", encoding="utf-8") as f:
                f.write(f"# Project Summary: {summary['name']}\n\n")
                f.write(f"**Generated by**: LibrarianBee\n")
                f.write(
                    f"## Overview\n- Path: {summary['path']}\n- Docs: {len(summary['documents'])}\n\n"
                )
                f.write(f"## Tech Stack\n- Lang: {summary['tech_stack']['language']}\n")
                f.write(f"## Docs\n")
                for doc in summary["documents"]:
                    f.write(f"- {doc['filename']}: {doc['summary']}\n")
            logger.info(f"📝 项目摘要已保存: {sp}")
        except Exception as e:
            logger.error(f"保存项目摘要失败: {e}")

    def _build_knowledge_index(self, target_path: Path, summary: Dict):
        idx = {
            "project_name": summary["name"],
            "project_path": str(target_path),
            "main_files": [f.name for f in self._find_key_files(target_path)[:5]],
            "tech_stack": summary["tech_stack"],
        }
        self._update_global_index(idx)
        logger.info(f"📚 知识索引已构建")

    def _find_key_files(self, target_path: Path) -> list:
        key_files = []
        for p in ["main.py", "app.py", "requirements.txt", "__init__.py"]:
            if (target_path / p).exists():
                key_files.append(target_path / p)
        return key_files

    def _update_global_index(self, knowledge_index: Dict):
        try:
            ip = self.settings.BASE_DIR / ".knowledge_index.json"
            existing = []
            if ip.exists():
                with open(ip, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(knowledge_index)
            with open(ip, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"更新索引失败: {e}")

    def get_library_status(self) -> Dict[str, Any]:
        return {
            "bee_type": "LibrarianBee",
            "total_projects": len(self.knowledge_index),
            "last_scan_time": "recent",
        }
