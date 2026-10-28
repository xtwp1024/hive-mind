# -*- coding: utf-8 -*-
"""
文件复制蜜蜂 - 简单的文件复制工具
功能：复制单个文件或批量复制文件
复杂度：低 (< 100行)
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from hive.bees.base_bee import BaseBee


class CopierBee(BaseBee):
    """负责文件复制操作的蜜蜂"""
    
    def __init__(self):
        super().__init__(
            name="copier_bee",
            role="file_copier",
            description="复制文件和目录"
        )
        self.copied_count = 0
        self.failed_count = 0
        self.copied_files = []
    
    def copy_file(self, source: str, destination: str) -> bool:
        """
        复制单个文件
        
        Args:
            source: 源文件路径
            destination: 目标文件路径
        
        Returns:
            bool: 是否成功
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            # 检查源文件是否存在
            if not source_path.exists():
                print(f"❌ 源文件不存在: {source}")
                self.failed_count += 1
                return False
            
            # 如果目标是目录，则在目录中使用源文件名
            if dest_path.is_dir():
                dest_path = dest_path / source_path.name
            
            # 执行复制
            shutil.copy2(source_path, dest_path)
            
            self.copied_count += 1
            self.copied_files.append({
                'source': str(source_path),
                'destination': str(dest_path)
            })
            
            print(f"✅ 已复制: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            print(f"❌ 复制失败: {source} -> {destination}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return False
    
    def copy_files(self, file_pairs: List[tuple]) -> dict:
        """
        批量复制文件
        
        Args:
            file_pairs: [(源文件, 目标文件), ...] 的列表
        
        Returns:
            dict: 复制结果统计
        """
        results = {
            'total': len(file_pairs),
            'copied': 0,
            'failed': 0,
            'files': []
        }
        
        for source, dest in file_pairs:
            if self.copy_file(source, dest):
                results['copied'] += 1
                results['files'].append({
                    'source': source,
                    'destination': dest,
                    'status': 'success'
                })
            else:
                results['failed'] += 1
                results['files'].append({
                    'source': source,
                    'destination': dest,
                    'status': 'failed'
                })
        
        return results
    
    def copy_directory(self, source: str, destination: str) -> bool:
        """
        复制整个目录
        
        Args:
            source: 源目录路径
            destination: 目标目录路径
        
        Returns:
            bool: 是否成功
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            if not source_path.exists() or not source_path.is_dir():
                print(f"❌ 源目录不存在: {source}")
                self.failed_count += 1
                return False
            
            # 复制整个目录树
            shutil.copytree(source_path, dest_path)
            
            self.copied_count += 1
            print(f"✅ 已复制目录: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            print(f"❌ 目录复制失败: {source} -> {destination}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return False
    
    def get_statistics(self) -> dict:
        """
        获取复制统计信息
        
        Returns:
            dict: 包含复制次数、失败次数等统计
        """
        return {
            'copied_count': self.copied_count,
            'failed_count': self.failed_count,
            'success_rate': (
                self.copied_count / (self.copied_count + self.failed_count) * 100
                if (self.copied_count + self.failed_count) > 0 else 100
            ),
            'recent_copies': self.copied_files[-5:] if self.copied_files else []
        }
    
    def execute(self, task: str, **kwargs) -> any:
        """
        执行复制任务
        
        Args:
            task: 任务类型 ('copy_file', 'copy_files', 'copy_directory')
            **kwargs: 任务参数
        """
        if task == 'copy_file':
            return self.copy_file(kwargs.get('source'), kwargs.get('destination'))
        
        elif task == 'copy_files':
            return self.copy_files(kwargs.get('file_pairs', []))
        
        elif task == 'copy_directory':
            return self.copy_directory(kwargs.get('source'), kwargs.get('destination'))
        
        else:
            print(f"❌ 未知任务类型: {task}")
            return None
