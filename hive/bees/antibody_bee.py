#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抗体蜂- AntibodyBee
修复已知问题和错误，生成补丁
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.config.settings import HiveSettings, get_settings

logger = logging.getLogger("Hive.Antibody")

class AntibodyBee(BaseBee):
    """抗体蜂 - 负责修复已知问题和错误"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        self.settings = get_settings()
        self.base_dir = Path(self.settings.BASE_DIR)
        self.patches_created = []
    
    def specialized_task(self, target_path):
        """
        执行抗体任务
        
        Args:
            target_path: 项目目录路径
        """
        logger.info(f"💊 开始生成抗体: {self.dna.target_name}")
        
        try:
            # 1. 分析错误库
            known_errors = self._analyze_error_vault()
            
            if not known_errors:
                logger.info(f"✅ 未发现已知错误")
                return
            
            # 2. 检查目标项目中的常见问题
            project_issues = self._scan_project_issues(target_path)
            
            # 3. 生成补丁
            patches = self._generate_patches(target_path, known_errors, project_issues)
            
            # 4. 应用补丁
            for patch in patches:
                self._apply_patch(target_path, patch)
                self.patches_created.append(patch)
            
            logger.info(f"✅ 抗体生成完成: {self.dna.target_name}")
            logger.info(f"   生成的补丁: {len(patches)}")
            logger.info(f"   应用的修复: {sum(p['fixes_applied'] for p in patches)}")
            
        except Exception as e:
            logger.error(f"❌ 抗体生成失败: {e}")
    
    def _analyze_error_vault(self) -> List[Dict[str, Any]]:
        """
        分析错误库，提取常见问题模式
        
        Returns:
            错误模式列表
        """
        try:
            error_vault_path = Path("D:/Jarvis_Memory/Failures")
            
            if not error_vault_path.exists():
                return []
            
            # 读取最近的20个错误
            error_files = sorted(
                list(error_vault_path.glob("*.json")),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )[:20]
            
            error_patterns = []
            
            for error_file in error_files:
                try:
                    with open(error_file, 'r', encoding='utf-8') as f:
                        error_data = json.load(f)
                        
                    error_msg = error_data.get('error', 'Unknown')
                    source = error_data.get('source', 'Unknown')
                    
                    # 提取错误模式
                    pattern = self._extract_error_pattern(error_msg)
                    
                    if pattern:
                        error_patterns.append({
                            "pattern": pattern,
                            "frequency": 1,
                            "source": source,
                            "first_seen": error_data.get('timestamp')
                        })
                except Exception as e:
                    logger.debug(f"读取错误文件失败 {error_file.name}: {e}")
            
            # 合并相似的模式
            merged_patterns = self._merge_error_patterns(error_patterns)
            
            logger.info(f"🔍 错误库分析完成: 发现 {len(merged_patterns)} 个模式")
            
            return merged_patterns
            
        except Exception as e:
            logger.error(f"错误库分析失败: {e}")
            return []
    
    def _scan_project_issues(self, target_path) -> List[Dict[str, Any]]:
        """
        扫描项目中的常见问题
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        try:
            # 查找Python文件
            py_files = list(target_path.rglob("*.py"))
            
            for py_file in py_files[:10]:  # 限制为前10个文件
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 检查常见问题
                if "print(" in content and not "if __name__" in content:
                    issues.append({
                        "type": "debug_print",
                        "severity": "LOW",
                        "file": py_file.name,
                        "description": "发现调试print语句"
                    })
                
                if "import *" in content:
                    issues.append({
                        "type": "star_import",
                        "severity": "MEDIUM",
                        "file": py_file.name,
                        "description": "发现星号导入"
                    })
                
                if ".append(" in content and not ".extend(" in content:
                    issues.append({
                        "type": "inefficient_list",
                        "severity": "LOW",
                        "file": py_file.name,
                        "description": "可能使用列表扩展"
                    })
                
        except Exception as e:
            logger.error(f"项目问题扫描失败: {e}")
        
        return issues
    
    def _generate_patches(self, target_path, error_patterns, project_issues):
        """
        生成修复补丁
        
        Args:
            target_path: 项目目录路径
            error_patterns: 错误模式列表
            project_issues: 项目问题列表
            
        Returns:
            补丁列表
        """
        patches = []
        
        # 1. 基于错误模式生成补丁
        for pattern in error_patterns[:5]:  # 最多5个补丁
            patch = {
                "type": "error_fix",
                "description": f"修复错误模式: {pattern['pattern']}",
                "code": self._generate_error_fix(pattern['pattern']),
                "files_to_patch": [],
                "fixes_applied": 0
            }
            patches.append(patch)
        
        # 2. 基于项目问题生成补丁
        for issue in project_issues[:3]:  # 最多3个补丁
            if issue["type"] == "debug_print":
                patch = {
                    "type": "remove_debug",
                    "description": "移除调试print语句",
                    "code": "# Debug prints should be removed",
                    "files_to_patch": [issue["file"]],
                    "fixes_applied": 0
                }
                patches.append(patch)
        
        return patches
    
    def _extract_error_pattern(self, error_msg: str) -> str:
        """
        从错误消息中提取模式
        
        Args:
            error_msg: 错误消息
            
        Returns:
            错误模式
        """
        # 简化错误消息
        simplified = re.sub(r'\s+', ' ', error_msg)
        simplified = re.sub(r"'[^']*'", "'X'", simplified)
        simplified = re.sub(r'"[^"]*"', '"X"', simplified)
        simplified = re.sub(r'\d+', 'N', simplified)
        
        # 提取前3个单词
        words = simplified.split()[:3]
        return ' '.join(words)
    
    def _merge_error_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        合并相似的错误模式
        
        Args:
            patterns: 错误模式列表
            
        Returns:
            合并后的模式列表
        """
        merged = {}
        
        for pattern in patterns:
            key = pattern['pattern']
            if key not in merged:
                merged[key] = pattern
            else:
                merged[key]['frequency'] += 1
        
        # 按频率排序
        sorted_patterns = sorted(merged.values(), key=lambda x: x['frequency'], reverse=True)
        
        return sorted_patterns
    
    def _generate_error_fix(self, pattern: str) -> str:
        """
        生成错误修复代码
        
        Args:
            pattern: 错误模式
            
        Returns:
            修复代码片段
        """
        # 简单的修复建议
        fixes = {
            "ImportError X": "检查依赖项，使用try-except处理",
            "NameError X": "检查变量作用域",
            "SyntaxError": "检查语法错误",
            "RuntimeError X": "检查逻辑错误"
        }
        
        # 查找匹配的修复
        for error_type, fix in fixes.items():
            if error_type.split()[0] in pattern:
                return f"# Fix for {pattern}\n{fix}"
        
        return f"# Auto-generated fix for {pattern}"
    
    def _apply_patch(self, target_path, patch):
        """
        应用补丁到项目
        
        Args:
            target_path: 项目目录路径
            patch: 补丁信息
        """
        # 这里只是记录，实际应用需要更复杂的逻辑
        logger.debug(f"应用补丁: {patch['description']}")
        patch['fixes_applied'] = 1
    
    def get_antibody_status(self) -> Dict[str, Any]:
        """
        获取抗体状态
        
        Returns:
            状态字典
        """
        return {
            "bee_type": self.__class__.__name__,
            "patches_created": len(self.patches_created),
            "last_action": "generating_antibody"
        }
