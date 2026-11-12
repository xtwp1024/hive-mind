#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArchivistBee (档案员蜂) - 本地资产整理与去重
"""

import json
import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger("Hive.Archivist")


class ArchivistBee:
    """
    档案员蜂 - 负责整理本地已下载项目并同步到去重系统

    职责:
    1. 扫描 D:\Inbound 目录
    2. 提取所有项目名称
    3. 将项目信息写入 discovery_history.json
    4. 防止重复下载
    """

    def __init__(self, inbound_dir: Path = Path(r"D:\Inbound")):
        self.inbound_dir = inbound_dir
        self.history_file = Path("discovery_history.json")
        logger.info(f"📚 档案员蜂已就位，监控目录: {self.inbound_dir}")

    def scan_local_projects(self) -> Set[str]:
        """
        扫描本地项目目录

        Returns:
            项目名称集合
        """
        if not self.inbound_dir.exists():
            logger.warning(f"⚠️ 目录不存在: {self.inbound_dir}")
            return set()

        projects = set()
        try:
            for item in self.inbound_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    projects.add(item.name)

            logger.info(f"📦 扫描到 {len(projects)} 个本地项目")
            return projects
        except Exception as e:
            logger.error(f"❌ 扫描失败: {e}")
            return set()

    def generate_urls_from_names(self, project_names: Set[str]) -> List[str]:
        """
        从项目名称生成 GitHub URL (用于去重)

        Args:
            project_names: 项目名称集合

        Returns:
            URL 列表
        """
        urls = []
        for name in project_names:
            # 生成标准 GitHub URL 格式用于去重
            # 实际 URL 可能不存在，但用于去重足够了
            url = f"https://github.com/unknown/{name}.git"
            urls.append(url)

        return urls

    def sync_to_history(self) -> int:
        """
        将本地项目同步到历史记录

        Returns:
            新增记录数量
        """
        # 1. 扫描本地项目
        local_projects = self.scan_local_projects()
        if not local_projects:
            logger.info("📭 无本地项目需要同步")
            return 0

        # 2. 生成 URL
        urls = self.generate_urls_from_names(local_projects)

        # 3. 读取现有历史
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                hist_data = {"assimilated_urls": []}
        else:
            hist_data = {"assimilated_urls": []}

        # 4. 去重并添加
        existing_urls = set(hist_data.get("assimilated_urls", []))
        new_urls = [u for u in urls if u not in existing_urls]

        if new_urls:
            hist_data["assimilated_urls"].extend(new_urls)

            # 5. 保存
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(hist_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 [档案同步] 新增 {len(new_urls)} 条历史记录")
            logger.info(f"📊 总历史记录: {len(hist_data['assimilated_urls'])} 条")
            return len(new_urls)
        else:
            logger.info("✅ 所有本地项目已在历史记录中")
            return 0


def run_archivist_sync():
    """运行档案员同步任务"""
    archivist = ArchivistBee()
    return archivist.sync_to_history()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_archivist_sync()
