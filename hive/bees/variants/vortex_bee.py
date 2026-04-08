#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涡流蜂 - VortexBee
深度代码审计和优化

变异率：13.00%
特殊能力：代码复杂度分析 + 重构建议
"""

import ast
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import collections
import time

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

try:
    from hive.utils.brain_interface import BrainInterface, get_brain
except ImportError:
    BrainInterface = None
    get_brain = None

logger = get_logger(__name__)

# --- Adopted from MetaGPT (v0.X) ---
# ref: D:\Inbound\MetaGPT\metagpt\actions\write_code_review.py

AUDIT_PROMPT_TEMPLATE = """
# System
Role: You are a professional software engineer (VortexBee), and your main task is to review and revise the code.
Goal: Ensure that the code conforms to the google-style standards, is elegantly designed and modularized, easy to read and maintain.
Language: Chinese (Simplified) for analysis, English for code symbols.

# Context
{context}

# Code to be Reviewed: {filename}
```python
{code}
```
"""

AUDIT_FORMAT_EXAMPLE = """
# Format Example
## Code Review: Ordered List
1. Is the code implemented as per the requirements?
2. Is the code logic completely correct?
3. Are all functions implemented?
4. ...

## Actions: Ordered List
1. Fix the `handle_events` method...
2. Implement function B...

## Code Review Result:
LGTM (Looks Good To Me) or LBTM (Looks Bad To Me)
"""
# -----------------------------------


class VortexBee(BaseBee):
    """
    涡流蜂 - 负责深度代码审计和优化建议
    
    特殊能力：
    - 计算代码复杂度
    - 识别代码异味
    - 生成重构建议
    - 分析代码质量指标
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.code_metrics = {}
        self.refactoring_suggestions = []
        self._brain: Optional[BrainInterface] = None

    def _get_brain(self) -> Optional[BrainInterface]:
        """懒加载 BrainInterface 单例"""
        if self._brain is None and BrainInterface is not None and get_brain is not None:
            self._brain = get_brain()
        return self._brain

    def _call_llm_audit(
        self,
        file_path: str,
        code_content: str,
        context: str = ""
    ) -> Optional[str]:
        """
        调用 Brain LLM 获取真实重构建议（同步版本）

        Args:
            file_path: 目标文件路径
            code_content: 代码内容
            context: 额外上下文

        Returns:
            LLM 生成的审计建议文本，失败返回 None
        """
        brain = self._get_brain()
        if brain is None:
            logger.warning("🌀 VortexBee: BrainInterface 不可用，跳过 LLM 调用")
            return None

        prompt = AUDIT_PROMPT_TEMPLATE.format(
            context=context or "No specific context provided.",
            filename=Path(file_path).name,
            code=code_content
        )

        system_prompt = """你是一位资深的软件工程师（VortexBee），负责代码审查和重构。
你的任务是：
1. 审查代码是否符合 Google 风格标准
2. 指出代码设计问题和潜在 Bug
3. 提供具体的重构建议和代码示例

请用中文回答，对每个问题给出：
- 问题描述
- 严重程度（高/中/低）
- 具体重构建议（包含代码示例如果适用）

只输出实质性问题，不要泛泛而谈。"""

        try:
            # BrainInterface.consult() 是同步的
            result = brain.consult(prompt, system_prompt)
            if result:
                logger.info(f"🧠 VortexBee LLM 审计完成: {Path(file_path).name}")
                return result
            else:
                logger.warning(f"🌀 VortexBee LLM 返回空结果: {file_path}")
                return None
        except Exception as e:
            logger.error(f"🌀 VortexBee LLM 调用失败: {e}")
            return None

    def _audit_file_with_llm(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        对单个文件执行 LLM 审计

        Returns:
            包含 refactoring_code_blocks 的字典，失败返回 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()

            # 只对有意义的文件调用 LLM（> 20 行）
            if len(code_content.split('\n')) < 20:
                return None

            # 检测问题上下文
            smells = self._detect_code_smells([{
                "path": str(file_path),
                "extension": ".py",
                "lines": len(code_content.split('\n'))
            }])

            issues = []
            if smells['long_functions']:
                issues.append(f"发现 {len(smells['long_functions'])} 个长函数")
            if smells['deep_nesting']:
                issues.append(f"发现 {len(smells['deep_nesting'])} 处深层嵌套")
            if smells['large_files']:
                issues.append(f"文件过大 ({smells['large_files'][0]['lines']} 行)")

            if not issues:
                return None  # 没有明显问题，不需要 LLM

            context = "检测到的问题: " + "; ".join(issues)
            llm_result = self._call_llm_audit(
                str(file_path), code_content, context
            )

            if llm_result:
                return {
                    "file": str(file_path),
                    "refactoring_code_blocks": llm_result,
                    "issues_found": issues
                }
        except Exception as e:
            logger.debug(f"🌀 文件 LLM 审计失败 {file_path}: {e}")

        return None
    
    def specialized_task(self, target_path):
        """
        执行深度代码审计任务

        Args:
            target_path: 项目目录路径
        """
        logger.info(f"🌀 VortexBee 开始任务: {self.dna.target_name}")

        try:
            # 1. 扫描所有代码文件
            code_files = self._scan_code_files(target_path)

            # 2. 分析代码复杂度
            complexity_analysis = self._analyze_complexity(code_files)

            # 3. 识别代码异味
            code_smells = self._detect_code_smells(code_files)

            # 4. 生成启发式重构建议（降级方案）
            heuristic_suggestions = self._generate_refactoring_suggestions(
                complexity_analysis, code_smells
            )

            # 5. 计算代码质量分数
            quality_score = self._calculate_quality_score(
                complexity_analysis, code_smells
            )

            # 6. ===== Phase 0 Live Link: 调用 LLM 获取真实重构建议 =====
            llm_audit_results: List[Dict[str, Any]] = []
            files_with_issues = (
                code_smells.get('long_functions', []) +
                code_smells.get('large_files', [])
            )

            if files_with_issues and BrainInterface is not None:
                logger.info(f"🧠 VortexBee 开始 LLM 审计 ({len(files_with_issues)} 个文件)...")

                # 收集需要审计的文件路径（使用绝对路径）
                target = Path(target_path)
                files_to_audit: List[Path] = []
                seen_paths: set = set()
                for smell in files_with_issues:
                    file_str = smell.get('file') or smell.get('path', '')
                    if file_str and file_str not in seen_paths:
                        # 优先尝试绝对路径，再尝试相对 target_path 的路径
                        p = Path(file_str)
                        if not p.is_absolute():
                            p = target / p
                        if p.exists():
                            files_to_audit.append(p)
                            seen_paths.add(file_str)

                # 限制 LLM 调用次数，避免过度消耗
                max_audit_files = 5
                for file_path in files_to_audit[:max_audit_files]:
                    logger.info(f"🌀 正在审计: {file_path.name}")
                    result = self._audit_file_with_llm(file_path)
                    if result:
                        llm_audit_results.append(result)
                    # 避免频率限制
                    time.sleep(1)

                logger.info(f"🧠 VortexBee LLM 审计完成: {len(llm_audit_results)} 个文件")

            # 7. 保存审计报告
            self._save_audit_report({
                "project": self.dna.target_name,
                "files_analyzed": len(code_files),
                "complexity_analysis": complexity_analysis,
                "code_smells": code_smells,
                "refactoring_suggestions": heuristic_suggestions,
                "llm_audit_results": llm_audit_results,  # Phase 0 Live Link 新增
                "quality_score": quality_score,
                "timestamp": __import__('time').ctime()
            })

            logger.info(f"✅ VortexBee 任务完成: {self.dna.target_name}")
            logger.info(f"   分析文件数: {len(code_files)}")
            logger.info(f"   代码质量分数: {quality_score:.2f}/100")
            logger.info(f"   启发式建议: {len(heuristic_suggestions)}")
            logger.info(f"   LLM 深度审计: {len(llm_audit_results)} 个文件")

            return {
                "status": "success",
                "files_analyzed": len(code_files),
                "quality_score": quality_score,
                "refactoring_suggestions": len(heuristic_suggestions),
                "llm_audit_files": len(llm_audit_results)
            }

        except Exception as e:
            logger.error(f"❌ VortexBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}

    def generate_audit_payload(self, file_path: str, context: str = "") -> dict:
        """
        生成 LLM 审计所需的 Prompt Payload (MetaGPT Style)
        
        Args:
            file_path: 目标文件路径
            context: 额外的上下文信息
            
        Returns:
            dict: 包含 prompt 和 meta 信息的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
                
            prompt = AUDIT_PROMPT_TEMPLATE.format(
                context=context or "No specific context provided.",
                filename=Path(file_path).name,
                code=code_content
            )
            
            return {
                "target_file": file_path,
                "prompt": prompt,
                "instruction": AUDIT_FORMAT_EXAMPLE,
                "model_suggestion": "gpt-4-turbo"  # MetaGPT usually requires GPT-4 class
            }
        except Exception as e:
            logger.error(f"生成审计 Payload 失败: {e}")
            return {}
    
    def _scan_code_files(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        扫描所有代码文件
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            代码文件列表
        """
        code_files = []
        code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go']
        
        for file_path in target_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in code_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    code_files.append({
                        "path": str(file_path.relative_to(target_path)),
                        "extension": file_path.suffix,
                        "size": len(content),
                        "lines": len(content.split('\n'))
                    })
                except Exception as e:
                    logger.debug(f"读取文件失败 {file_path}: {e}")
        
        return code_files
    
    def _analyze_complexity(self, code_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析代码复杂度
        
        Args:
            code_files: 代码文件列表
            
        Returns:
            复杂度分析结果
        """
        complexity_metrics = {
            "total_files": len(code_files),
            "total_lines": 0,
            "avg_lines_per_file": 0,
            "complex_files": [],
            "simple_files": [],
            "complexity_distribution": {
                "low": 0,      # < 100 行
                "medium": 0,   # 100-500 行
                "high": 0,     # 500-1000 行
                "very_high": 0 # > 1000 行
            }
        }
        
        for file_info in code_files:
            lines = file_info['lines']
            complexity_metrics['total_lines'] += lines
            
            # 分类复杂度
            if lines < 100:
                complexity_metrics['complexity_distribution']['low'] += 1
                complexity_metrics['simple_files'].append(file_info['path'])
            elif lines < 500:
                complexity_metrics['complexity_distribution']['medium'] += 1
            elif lines < 1000:
                complexity_metrics['complexity_distribution']['high'] += 1
                complexity_metrics['complex_files'].append(file_info['path'])
            else:
                complexity_metrics['complexity_distribution']['very_high'] += 1
                complexity_metrics['complex_files'].append(file_info['path'])
        
        # 计算平均值
        if complexity_metrics['total_files'] > 0:
            complexity_metrics['avg_lines_per_file'] = \
                complexity_metrics['total_lines'] / complexity_metrics['total_files']
        
        return complexity_metrics
    
    def _detect_code_smells(self, code_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检测代码异味
        
        Args:
            code_files: 代码文件列表
            
        Returns:
            代码异味检测结果
        """
        smells = {
            "long_functions": [],
            "deep_nesting": [],
            "duplicate_code": [],
            "magic_numbers": [],
            "large_files": [],
            "long_lines": []
        }
        
        for file_info in code_files:
            if file_info['extension'] != '.py':
                continue  # 目前只分析Python文件
            
            file_path = Path(file_info['path'])
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 检测长函数（通过AST分析）
                smells['long_functions'].extend(
                    self._detect_long_functions(content, file_info['path'])
                )
                
                # 检测深层嵌套
                smells['deep_nesting'].extend(
                    self._detect_deep_nesting(content, file_info['path'])
                )
                
                # 检测魔法数字
                smells['magic_numbers'].extend(
                    self._detect_magic_numbers(content, file_info['path'])
                )
                
                # 检测大文件
                if file_info['lines'] > 500:
                    smells['large_files'].append({
                        "file": file_info['path'],
                        "lines": file_info['lines']
                    })
                
                # 检测长行
                smells['long_lines'].extend(
                    self._detect_long_lines(content, file_info['path'])
                )
                
            except Exception as e:
                logger.debug(f"检测代码异味失败 {file_info['path']}: {e}")
        
        return smells
    
    def _detect_long_functions(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        检测长函数
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            长函数列表
        """
        long_functions = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 计算函数行数
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    func_lines = end_line - start_line + 1
                    
                    if func_lines > 50:  # 超过50行视为长函数
                        long_functions.append({
                            "file": file_path,
                            "function": node.name,
                            "lines": func_lines,
                            "line_number": start_line
                        })
        except SyntaxError:
            pass  # 忽略无法解析的文件
        
        return long_functions
    
    def _detect_deep_nesting(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        检测深层嵌套
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            深层嵌套列表
        """
        deep_nesting = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 计算缩进级别
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                indent_level = (len(line) - len(stripped)) // 4  # 假设4空格缩进
                
                if indent_level > 4:  # 超过4层嵌套
                    deep_nesting.append({
                        "file": file_path,
                        "line": i,
                        "indent_level": indent_level,
                        "content": stripped[:50]  # 只保存前50个字符
                    })
        
        return deep_nesting
    
    def _detect_magic_numbers(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        检测魔法数字
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            魔法数字列表
        """
        magic_numbers = []
        
        # 匹配数字（排除常见的0, 1, 2等）
        pattern = r'\b(?!0|1|2|10|100|1000)\d{2,}\b'
        
        for match in re.finditer(pattern, content):
            # 检查是否在注释或字符串中
            start = match.start()
            # 简单检查：如果在引号内，跳过
            line_start = content.rfind('\n', 0, start) + 1
            line_end = content.find('\n', start)
            if line_end == -1:
                line_end = len(content)
            line = content[line_start:line_end]
            
            # 如果在引号内或注释中，跳过
            if '#' in line and line.index('#') < start - line_start:
                continue
            
            magic_numbers.append({
                "file": file_path,
                "number": match.group(),
                "line": content[:start].count('\n') + 1
            })
        
        return magic_numbers[:50]  # 限制数量
    
    def _detect_long_lines(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        检测长行
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            长行列表
        """
        long_lines = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # 超过120字符
                long_lines.append({
                    "file": file_path,
                    "line": i,
                    "length": len(line)
                })
        
        return long_lines[:20]  # 限制数量
    
    def _generate_refactoring_suggestions(
        self, 
        complexity: Dict[str, Any], 
        smells: Dict[str, Any]
    ) -> List[str]:
        """
        生成重构建议
        
        Args:
            complexity: 复杂度分析结果
            smells: 代码异味检测结果
            
        Returns:
            重构建议列表
        """
        suggestions = []
        
        # 基于复杂度的建议
        high_complexity = complexity['complexity_distribution']['high'] + \
                         complexity['complexity_distribution']['very_high']
        
        if high_complexity > 0:
            suggestions.append(
                f"发现 {high_complexity} 个高复杂度文件，建议拆分为多个模块"
            )
        
        # 基于代码异味的建议
        if len(smells['long_functions']) > 0:
            suggestions.append(
                f"发现 {len(smells['long_functions'])} 个长函数，建议提取子函数"
            )
        
        if len(smells['deep_nesting']) > 0:
            suggestions.append(
                f"发现 {len(smells['deep_nesting'])} 处深层嵌套，建议使用提前返回减少嵌套"
            )
        
        if len(smells['large_files']) > 0:
            suggestions.append(
                f"发现 {len(smells['large_files'])} 个大文件，建议按功能拆分"
            )
        
        if not suggestions:
            suggestions.append("代码质量良好，无需紧急重构")
        
        return suggestions
    
    def _calculate_quality_score(
        self, 
        complexity: Dict[str, Any], 
        smells: Dict[str, Any]
    ) -> float:
        """
        计算代码质量分数
        
        Args:
            complexity: 复杂度分析结果
            smells: 代码异味检测结果
            
        Returns:
            质量分数 (0-100)
        """
        score = 100.0
        
        # 基于复杂度的扣分
        high_complexity_ratio = (
            complexity['complexity_distribution']['high'] + 
            complexity['complexity_distribution']['very_high']
        ) / max(complexity['total_files'], 1)
        score -= high_complexity_ratio * 30
        
        # 基于代码异味的扣分
        total_smells = (
            len(smells['long_functions']) +
            len(smells['deep_nesting']) +
            len(smells['large_files'])
        )
        smell_penalty = min(total_smells * 2, 20)  # 最多扣20分
        score -= smell_penalty
        
        # 确保分数在0-100范围内
        return max(0.0, min(100.0, score))
    
    def _save_audit_report(self, report_data: Dict[str, Any]):
        """
        保存审计报告
        
        Args:
            report_data: 报告数据
        """
        try:
            import json
            reports_dir = Path("vortex_reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_filename = f"{self.dna.target_name.replace('/', '_')}_vortex_report.json"
            report_path = reports_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 代码审计报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存审计报告失败: {e}")
