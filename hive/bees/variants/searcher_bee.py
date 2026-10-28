# -*- coding: utf-8 -*-
"""
SearcherBee - 内容搜索蜂
简单的文件内容搜索功能
"""

import re
from pathlib import Path
from typing import Dict, Any, List
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

class SearcherBee(BaseBee):
    """
    搜索者蜂 - 内容搜索工具
    
    能力：
    1. 搜索文件中的关键词
    2. 使用正则表达式匹配
    3. 生成搜索结果报告
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("SearcherBee")
    
    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行搜索任务
        
        Args:
            target: 目标路径
            
        Returns:
            搜索结果字典
        """
        # 从DNA中获取搜索参数
        keyword = getattr(self.dna, 'search_keyword', '')
        use_regex = getattr(self.dna, 'use_regex', False)
        
        if not keyword:
            return {
                "status": "error",
                "message": "未指定搜索关键词"
            }
        
        self.logger.info(f"🔍 Searcher 开始搜索: {keyword} in {target}")
        
        if not target.exists():
            return {
                "status": "error",
                "message": f"目标不存在: {target}"
            }
        
        if target.is_file():
            return self._search_file(target, keyword, use_regex)
        
        return self._search_directory(target, keyword, use_regex)
    
    def _search_file(self, file_path: Path, keyword: str, use_regex: bool) -> Dict[str, Any]:
        """
        在单个文件中搜索
        
        Args:
            file_path: 文件路径
            keyword: 搜索关键词
            use_regex: 是否使用正则表达式
            
        Returns:
            搜索结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            matches = []
            search_pattern = keyword if use_regex else re.escape(keyword)
            pattern = re.compile(search_pattern, re.IGNORECASE)
            
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    matches.append({
                        "line": i,
                        "content": line.strip()[:200],
                        "match_count": len(pattern.findall(line))
                    })
            
            return {
                "status": "success",
                "target_type": "file",
                "target": str(file_path),
                "keyword": keyword,
                "use_regex": use_regex,
                "matches": matches,
                "total_matches": len(matches),
                "message": f"在文件中找到 {len(matches)} 处匹配"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"搜索文件失败: {e}"
            }
    
    def _search_directory(self, directory: Path, keyword: str, use_regex: bool) -> Dict[str, Any]:
        """
        在目录中搜索
        
        Args:
            directory: 目录路径
            keyword: 搜索关键词
            use_regex: 是否使用正则表达式
            
        Returns:
            搜索结果
        """
        results = []
        total_matches = 0
        files_with_matches = 0
        
        search_pattern = keyword if use_regex else re.escape(keyword)
        pattern = re.compile(search_pattern, re.IGNORECASE)
        
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
                
            # 跳过二进制文件
            if file_path.suffix in ['.pyc', '.so', '.dll', '.exe', '.bin']:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                matches = []
                for i, line in enumerate(lines, 1):
                    if pattern.search(line):
                        matches.append({
                            "line": i,
                            "content": line.strip()[:200]
                        })
                
                if matches:
                    files_with_matches += 1
                    total_matches += len(matches)
                    results.append({
                        "file": str(file_path.relative_to(directory)),
                        "matches": matches[:20]  # 限制每个文件的匹配数
                    })
                    
            except Exception as e:
                self.logger.debug(f"跳过文件 {file_path}: {e}")
        
        return {
            "status": "success",
            "target_type": "directory",
            "target": str(directory),
            "keyword": keyword,
            "use_regex": use_regex,
            "files_with_matches": files_with_matches,
            "total_matches": total_matches,
            "results": results[:50],  # 限制结果数量
            "message": f"在 {files_with_matches} 个文件中找到 {total_matches} 处匹配"
        }
    
    def search_multiple_keywords(self, directory: Path, keywords: List[str]) -> Dict[str, Any]:
        """
        搜索多个关键词
        
        Args:
            directory: 目录路径
            keywords: 关键词列表
            
        Returns:
            批量搜索结果
        """
        all_results = {}
        
        for keyword in keywords:
            result = self._search_directory(directory, keyword, False)
            all_results[keyword] = {
                "files_with_matches": result["files_with_matches"],
                "total_matches": result["total_matches"]
            }
        
        return {
            "status": "success",
            "keyword": keywords,
            "results": all_results,
            "message": f"完成 {len(keywords)} 个关键词的搜索"
        }
