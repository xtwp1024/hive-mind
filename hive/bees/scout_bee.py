#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Scout Bee (侦查蜂)
源自 Hive 1.0 Scout Agent，负责主动互联网侦查。
"""

import time
import json
import random
import urllib.parse
import urllib.request
import logging
from pathlib import Path
from typing import List, Dict, Any

from hive.core.dna import BeeDNA
from hive.config.settings import get_settings

# 第三方依赖
try:
    import requests
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
except ImportError:
    pass # 依赖将在运行时检查

logger = logging.getLogger("Hive.Scout")

class ScoutBee:
    """
    侦查蜂 - 互联网资源搜索者
    
    能力:
    - Google/DuckDuckGo 搜索
    - Anti-403 规避 (UserAgent轮询, Jitter)
    - 结果解析与去重
    """
    
    def __init__(self, dna: BeeDNA, result_queue=None):
        self.dna = dna
        self.id = dna.id
        self.result_queue = result_queue
        self.logger = logging.getLogger(f"Scout-{self.id}")
        self.settings = get_settings()
        
        self.loot_file = Path("scout_loot.json")
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        ]

    def specialized_task(self, target_path: Path) -> Dict[str, Any]:
        """
        执行侦查任务
        Target Path 在这里被解释为搜索关键词 (如果不是路径)
        或者如果提供了 brain_override，则使用其中的指令
        """
        try:
            # 检查依赖
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "Missing dependencies: requests, beautifulsoup4, fake-useragent"}

        # 确定搜索关键词
        # 1. 优先使用 DNA 中的 target_name
        # 2. 其次尝试读取 brain_override (为了演示，这里简化直接用 target)
        query = self.dna.target_name
        
        if not query or query == "None":
            query = "python agent framework" # 默认 fallback

        self.logger.info(f"🔭 开始全网侦查: '{query}'")
        
        results = self.sniff_robust(query)
        
        # 将结果写入 loot file (供 AutoFeast 消费)
        new_count = self._save_results(results, query)
        
        return {
            "scout_action": "search",
            "query": query,
            "found_count": len(results),
            "new_artifacts": new_count,
            "top_results": results[:3]
        }

    def sniff_robust(self, query: str, attempt=1, max_retries=3) -> List[Dict[str, str]]:
        """
        鲁棒搜索 (移植自 Scout Agent)
        """
        # Jitter
        time.sleep(random.uniform(2, 5))

        encoded_q = urllib.parse.quote(query)
        # 使用 Google
        url = f"https://www.google.com/search?q={encoded_q}&num=10"
        
        try:
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://www.google.com/",
                "DNT": "1"
            }
            
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=20)
            
            # 简单的 403 处理
            if response.status_code == 403 or response.status_code == 429:
                if attempt < max_retries:
                    self.logger.warning(f"⚠️ 触发防御 ({response.status_code})，正在重试...")
                    time.sleep(random.uniform(5, 10))
                    return self.sniff_robust(query, attempt + 1, max_retries)
                return []
                
            response.raise_for_status()
            
            # 解析
            soup = BeautifulSoup(response.text, "html.parser")
            results = self._extract_results_soup(soup)
            
            return results

        except Exception as e:
            self.logger.error(f"❌ 侦查失败: {e}")
            return []

    def _extract_results_soup(self, soup) -> List[Dict[str, str]]:
        """解析 HTML 提取链接"""
        results = []
        # Strategy 1: div.g
        for g in soup.find_all('div', class_='g'):
            title_node = g.find('h3')
            link_node = g.find('a', href=True)
            
            if title_node and link_node:
                title = title_node.get_text(strip=True)
                link = link_node['href']
                if link.startswith("/url?q="):
                    try:
                        link = link.split("/url?q=")[1].split("&")[0]
                        link = urllib.parse.unquote(link)
                    except (IndexError, AttributeError): pass
                
                if link.startswith("http"):
                    results.append({"title": title, "url": link})
        return results

    def _save_results(self, results: List[Dict], source_query: str) -> int:
        """保存结果到公共 loot 文件"""
        new_count = 0
        try:
            data = {"targets": []}
            if self.loot_file.exists():
                with open(self.loot_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            existing_urls = {t.get('url') for t in data['targets']}
            
            for r in results:
                if r['url'] not in existing_urls:
                    data['targets'].append({
                        "id": r['title'][:50],
                        "url": r['url'],
                        "type": "scout_finding",
                        "source_query": source_query,
                        "discovered_at": time.time()
                    })
                    new_count += 1
            
            if new_count > 0:
                with open(self.loot_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 已归档 {new_count} 个新发现!")
                
        except Exception as e:
            self.logger.error(f"保存 Loot 失败: {e}")
            
        return new_count
