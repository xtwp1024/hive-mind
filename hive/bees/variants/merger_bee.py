# -*- coding: utf-8 -*-
"""
文件合并蜜蜂 - 合并多个文件
功能：将多个文件合并为一个文件
复杂度：低 (< 100行)
"""

import os
from pathlib import Path
from typing import List, Optional
from hive.bees.base_bee import BaseBee


class MergerBee(BaseBee):
    """负责文件合并操作的蜜蜂"""
    
    def __init__(self):
        super().__init__(
            name="merger_bee",
            role="file_merger",
            description="合并多个文件为一个文件"
        )
        self.merged_count = 0
        self.failed_count = 0
        self.merged_files = []
    
    def merge_files(self, source_files: List[str], output_file: str, 
                   separator: str = "\n") -> bool:
        """
        合并多个文件到一个文件
        
        Args:
            source_files: 源文件路径列表
            output_file: 输出文件路径
            separator: 文件之间的分隔符
        
        Returns:
            bool: 是否成功
        """
        try:
            output_path = Path(output_file)
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            merged_content = []
            missing_files = []
            
            # 读取所有源文件
            for source_file in source_files:
                source_path = Path(source_file)
                
                if not source_path.exists():
                    missing_files.append(source_file)
                    continue
                
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    merged_content.append(content)
            
            # 如果有缺失的文件，警告但继续
            if missing_files:
                print(f"⚠️  以下文件不存在，已跳过: {missing_files}")
            
            if not merged_content:
                print(f"❌ 没有可合并的文件")
                self.failed_count += 1
                return False
            
            # 写入合并后的内容
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(separator.join(merged_content))
            
            self.merged_count += 1
            self.merged_files.append({
                'source_files': source_files,
                'output_file': output_file,
                'file_count': len(merged_content)
            })
            
            print(f"✅ 已合并 {len(merged_content)} 个文件到: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ 合并失败: {output_file}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return False
    
    def merge_text_files(self, directory: str, pattern: str = "*.txt",
                        output_file: str = "merged.txt") -> bool:
        """
        合并目录中匹配模式的所有文本文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            output_file: 输出文件名
        
        Returns:
            bool: 是否成功
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists() or not dir_path.is_dir():
                print(f"❌ 目录不存在: {directory}")
                self.failed_count += 1
                return False
            
            # 查找所有匹配的文件
            files = list(dir_path.glob(pattern))
            
            if not files:
                print(f"❌ 在 {directory} 中没有找到匹配 {pattern} 的文件")
                self.failed_count += 1
                return False
            
            # 获取完整路径
            file_paths = [str(f) for f in files]
            
            # 构建输出路径
            output_path = dir_path / output_file
            
            return self.merge_files(file_paths, str(output_path))
            
        except Exception as e:
            print(f"❌ 目录文件合并失败: {directory}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return False
    
    def merge_lines(self, source_files: List[str], output_file: str,
                   skip_empty: bool = True) -> bool:
        """
        合并文件的所有行
        
        Args:
            source_files: 源文件路径列表
            output_file: 输出文件路径
            skip_empty: 是否跳过空行
        
        Returns:
            bool: 是否成功
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            lines = []
            for source_file in source_files:
                source_path = Path(source_file)
                
                if not source_path.exists():
                    continue
                
                with open(source_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if skip_empty and not line.strip():
                            continue
                        lines.append(line)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            self.merged_count += 1
            print(f"✅ 已合并 {len(lines)} 行到: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ 行合并失败: {output_file}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return False
    
    def get_statistics(self) -> dict:
        """
        获取合并统计信息
        
        Returns:
            dict: 包含合并次数、失败次数等统计
        """
        return {
            'merged_count': self.merged_count,
            'failed_count': self.failed_count,
            'success_rate': (
                self.merged_count / (self.merged_count + self.failed_count) * 100
                if (self.merged_count + self.failed_count) > 0 else 100
            ),
            'recent_merges': self.merged_files[-5:] if self.merged_files else []
        }
    
    def execute(self, task: str, **kwargs) -> any:
        """
        执行合并任务
        
        Args:
            task: 任务类型 ('merge_files', 'merge_text_files', 'merge_lines')
            **kwargs: 任务参数
        """
        if task == 'merge_files':
            return self.merge_files(
                kwargs.get('source_files', []),
                kwargs.get('output_file', 'merged.txt'),
                kwargs.get('separator', '\n')
            )
        
        elif task == 'merge_text_files':
            return self.merge_text_files(
                kwargs.get('directory', '.'),
                kwargs.get('pattern', '*.txt'),
                kwargs.get('output_file', 'merged.txt')
            )
        
        elif task == 'merge_lines':
            return self.merge_lines(
                kwargs.get('source_files', []),
                kwargs.get('output_file', 'merged_lines.txt'),
                kwargs.get('skip_empty', True)
            )
        
        else:
            print(f"❌ 未知任务类型: {task}")
            return None
