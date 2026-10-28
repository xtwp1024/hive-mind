#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建筑蜂- ArchitectBee
重构和优化下载的代码，应用设计模式
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)

class ArchitectBee(BaseBee):
    """建筑蜂 - 负责代码重构和优化"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        self.report = {
            "code_smells": [], "design_patterns_applied": [],
            "refactored_files": [], "performance_improvements": []
        }
    
    def specialized_task(self, target_path: Path):
        logger.info(f"🏗️ 开始建筑: {self.dna.target_name}")
        try:
            py_files = list(target_path.rglob("*.py"))
            if not py_files: return
            
            for py_file in py_files[:10]: self._optimize_single_file(py_file)
            self._apply_project_level_refactoring(target_path)
            self._write_architect_report(target_path)
            
            logger.info(f"✅ 建筑完成: {len(py_files)} 文件分析")
        except Exception as e:
            logger.error(f"❌ 建筑失败: {e}")

    def _optimize_single_file(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = original = f.read()
            
            content = self._fix_long_lines(content, file_path.name)
            content = self._optimize_imports(content)
            content = self._add_type_hints(content)
            
            if content != original:
                with open(file_path, 'w', encoding='utf-8') as f: f.write(content)
                self.report["refactored_files"].append(file_path.name)
        except Exception as e:
            logger.error(f"优化文件失败 {file_path.name}: {e}")

    def _fix_long_lines(self, content: str, filename: str) -> str:
        lines, new_lines = content.split('\n'), []
        for line in lines:
            if len(line) > 100:
                self._handle_long_line(line, new_lines, filename)
            else:
                new_lines.append(line)
        return '\n'.join(new_lines)

    def _handle_long_line(self, line: str, new_lines: List[str], filename: str):
        if '(' in line and ')' in line:
            pos = line.rfind(')')
            if pos + 1 < len(line):
                new_lines.extend([line[:pos+1], line[pos+1:]])
                self.report["code_smells"].append({"type": "long_line", "file": filename, "severity": "low"})
                return
        new_lines.append(line)

    def _optimize_imports(self, content: str) -> str:
        pattern = re.compile(r'^import\s+([^\n]+)', re.MULTILINE)
        imports = pattern.findall(content)
        if len(imports) <= 10:
            return content

        std, third = [], []
        for i in imports:
            (std if i.split()[0].strip(',') in {'os', 'sys', 'json', 're', 'time', 'path', 'typing'} else third).append(i)

        if not third:
            return content

        new_imp = ['# Standard'] + std + ['\n# Third Party'] + third
        replacement = '\n'.join(new_imp) + '\n'

        # 只替换第一处匹配，然后删除其余所有 import 行，避免 N 次重复插入
        first = True
        def _replacer(m):
            nonlocal first
            if first:
                first = False
                return replacement
            return ''

        self.report["design_patterns_applied"].append("import_grouping")
        return pattern.sub(_replacer, content)

    def _add_type_hints(self, content: str) -> str:
        if 'from typing import' in content:
            return content

        funcs = re.findall(r'def\s+(\w+)\s*\((.*?)\)\s*:', content)
        for name, params in funcs[:5]:
            # 精确匹配完整函数签名，保留参数列表
            sig_pattern = re.compile(
                r'(def\s+' + re.escape(name) + r'\s*\(.*?\)\s*)(:)',
                re.DOTALL,
            )
            def _add_return(m):
                head = m.group(1)
                if '->' not in head:
                    self.report["performance_improvements"].append("added_type_hints")
                    return head + ' -> Any:'
                return m.group(0)
            content = sig_pattern.sub(_add_return, content, count=1)
        return content

    def _apply_project_level_refactoring(self, target_path: Path):
        self._create_init_file(target_path)
        self._create_setup_file(target_path)

    def _create_init_file(self, target_path: Path):
        init_file = target_path / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w', encoding='utf-8') as f: f.write('"""Initialize package."""\n')
            self.report["design_patterns_applied"].append("created_init")

    def _create_setup_file(self, target_path: Path):
        if list(target_path.glob("*.py")) and not (target_path / "setup.py").exists():
            with open(target_path / "setup.py", 'w', encoding='utf-8') as f:
                f.write(f'from setuptools import setup, find_packages\n\nsetup(\n    name="{self.dna.target_name}",\n    version="1.0.0",\n    packages=find_packages()\n)\n')
            self.report["refactored_files"].append("setup.py")

    def _write_architect_report(self, target_path: Path):
        r = self.report
        c = f"# Architect Report: {self.dna.target_name}\n\n"
        c += f"- Code Smells: {len(r['code_smells'])}\n"
        c += f"- Patterns Applied: {len(r['design_patterns_applied'])}\n"
        c += f"- Files Refactored: {len(r['refactored_files'])}\n"
        
        try:
            with open(target_path / "ARCHITECT_REPORT.md", 'w', encoding='utf-8') as f: f.write(c)
        except Exception: pass
