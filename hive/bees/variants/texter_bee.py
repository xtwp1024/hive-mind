# -*- coding: utf-8 -*-
"""
TexterBee - 文本处理蜂
简单的文本文件处理功能
"""

from pathlib import Path
from typing import Dict, Any, List
from hive.bees.base_bee import BaseBee
from hive.utils.logger import get_logger

class TexterBee(BaseBee):
    """
    文本蜂 - 文本处理工具
    
    能力：
    1. 读取文本文件
    2. 简单的文本转换
    3. 生成文本摘要
    """
    
    def __init__(self, bee_dna: Dict[str, Any], id: str, result_queue):
        super().__init__(bee_dna, id, result_queue)
        self.logger = get_logger("TexterBee")
    
    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行文本处理任务
        
        Args:
            target: 目标路径
            
        Returns:
            处理结果字典
        """
        self.logger.info(f"📝 Texter 开始处理: {target}")
        
        if not target.exists() or not target.is_file():
            return {
                "status": "error",
                "message": f"目标不是有效的文件: {target}"
            }
        
        # 读取文件内容
        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            analysis = self._analyze_text(content)
            
            return {
                "status": "success",
                "target": str(target),
                "analysis": analysis,
                "message": f"文本处理完成: {analysis['total_chars']} 字符"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"读取文件失败: {e}"
            }
    
    def _analyze_text(self, content: str) -> Dict[str, Any]:
        """
        分析文本内容
        
        Args:
            content: 文本内容
            
        Returns:
            分析结果
        """
        lines = content.split('\n')
        words = content.split()
        
        # 基本统计
        total_chars = len(content)
        total_lines = len(lines)
        total_words = len(words)
        empty_lines = sum(1 for line in lines if not line.strip())
        
        # 单词统计
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # 获取最常见的单词
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 摘要（前500字符）
        summary = content[:500] + "..." if len(content) > 500 else content
        
        return {
            "total_chars": total_chars,
            "total_lines": total_lines,
            "total_words": total_words,
            "empty_lines": empty_lines,
            "avg_words_per_line": round(total_words / total_lines, 2) if total_lines > 0 else 0,
            "unique_words": len(word_counts),
            "top_words": top_words,
            "summary": summary
        }
    
    def extract_keywords(self, content: str, min_length: int = 4) -> List[str]:
        """
        提取关键词
        
        Args:
            content: 文本内容
            min_length: 最小单词长度
            
        Returns:
            关键词列表
        """
        words = content.split()
        keywords = []
        
        word_counts = {}
        for word in words:
            word_clean = ''.join(c for c in word if c.isalpha()).lower()
            if len(word_clean) >= min_length:
                word_counts[word_clean] = word_counts.get(word_clean, 0) + 1
        
        # 返回出现频率最高的词
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        return [word for word, count in top_keywords]
