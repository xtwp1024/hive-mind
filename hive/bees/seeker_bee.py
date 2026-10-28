#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
寻蜂- SeekerBee
扫描下载的代码中的新GitHub URL，扩展蜂巢能力
"""

import logging
import re
import os
import json
import multiprocessing
from typing import Set, Any
from pathlib import Path
from time import ctime

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.config.settings import HiveSettings, get_settings
from hive.utils.logger import get_logger

logger = get_logger(__name__)

class SeekerBee(BaseBee):
    """寻蜂 - 扫描下载的代码中的新GitHub URL"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        self.settings = get_settings()
        self.found_urls = set()
    
    def specialized_task(self, target_path):
        """
        扫描目标路径，寻找新的GitHub项目URL
        
        Args:
            target_path: 项目目录路径
        """
        logger.info(f"🔍 开始扫描: {self.dna.target_name}")
        
        try:
            # 定义搜索模式
            github_pattern = re.compile(r'https://github\.com/[\w-]+/[\w.-]+')
            
            found_urls = set()
            scanned_files = 0
            
            # 遍历项目文件
            for root, dirs, files in os.walk(str(target_path)):
                for file in files:
                    # 只检查Python、Markdown和JavaScript文件
                    if file.endswith(('.py', '.md', '.js', '.json', '.yaml', '.yml')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # 查找所有GitHub URL
                                matches = github_pattern.findall(content)
                                found_urls.update(matches)
                                scanned_files += 1
                        except Exception as e:
                            logger.debug(f"扫描文件失败 {file}: {e}")
            
            self.found_urls = found_urls
            
            # 统计
            logger.info(f"📊 扫描完成: {self.dna.target_name}")
            logger.info(f"   扫描文件数: {scanned_files}")
            logger.info(f"   发现URL数: {len(found_urls)}")
            
            if found_urls:
                # 显示前10个发现的URL
                logger.info(f"   发现的URL示例:")
                for i, url in enumerate(list(found_urls)[:10], 1):
                    logger.info(f"     {i}. {url}")
                
                # 去重：排除已知URL
                unique_urls = self._deduplicate_urls(found_urls)
                logger.info(f"   去重后: {len(unique_urls)} 个唯一URL")
                
                # 保存结果
                self._save_scan_results(unique_urls)
                
                logger.info(f"✅ 扫描成功，发现 {len(unique_urls)} 个新项目")
            else:
                logger.warning(f"⚠️ 未发现新的GitHub URL")
                
        except Exception as e:
            logger.error(f"❌ 扫描失败: {e}")
            return {"error": str(e)}

        return {
            "found_urls": list(self.found_urls),
            "scanned_files": scanned_files,
            "new_urls_count": len(found_urls)
        }
    
    def _deduplicate_urls(self, urls: Set[str]) -> Set[str]:
        """
        去重URL，排除已知项目
        
        Args:
            urls: 发现的URL集合
            
        Returns:
            去重后的URL集合
        """
        try:
            history_path = self.settings.BASE_DIR / ".discovery_history"
            
            known_urls = set()
            if history_path.exists():
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    for item in history.get('assimilated_urls', []):
                        known_urls.add(item)
            
            logger.debug(f"已知URL数: {len(known_urls)}")
            
            # 过滤掉已知的URL
            new_urls = urls - known_urls
            
            logger.info(f"🔄 URL去重: {len(urls)} -> {len(new_urls)}")
            return new_urls
            
        except Exception as e:
            logger.error(f"URL去重失败: {e}")
            return urls
    
    def _save_scan_results(self, urls: Set[str]):
        """
        保存扫描结果到文件
        
        Args:
            urls: 发现的URL集合
        """
        try:
            results_path = self.settings.BASE_DIR / "seeker_results.json"
            
            results = {
                "scan_time": ctime(),
                "source_project": self.dna.target_name,
                "found_urls_count": len(urls),
                "urls": list(urls)
            }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 结果已保存: {results_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
