# -*- coding: utf-8 -*-
"""
蜂群吸收汇报

每小时生成一次汇报，保存到 Obsidian 格式。
"""

import datetime
import os
from pathlib import Path
from typing import Any

from hive.core.swarm import Swarm


class SwarmReporter:
    """蜂群汇报生成器"""

    def __init__(self, swarm: Swarm, output_dir: str = None):
        self.swarm = swarm

        # 默认输出到项目根目录的 obsidian_notes/swarm_reports
        if output_dir is None:
            base = Path(__file__).parent.parent.parent.parent
            output_dir = base / "obsidian_notes" / "swarm_reports"

        self.output_dir = Path(output_dir)

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self) -> str:
        """生成汇报"""

        now = datetime.datetime.now()
        gen = self.swarm.generation

        # 统计蜂种
        species_count = {}
        for atom in self.swarm.atoms.values():
            if atom.is_alive and atom.dna:
                role = atom.dna.role or "worker"
                species_count[role] = species_count.get(role, 0) + 1

        # 统计知识
        kb_stats = self.swarm.knowledge_base.get_stats()

        # 预计算值
        alive_count = len([a for a in self.swarm.atoms.values() if a.is_alive])
        active_count = len([a for a in self.swarm.atoms.values() if a.state.value == "active"])
        avg_fitness = self.swarm.metrics.avg_fitness
        max_fitness = self.swarm.metrics.max_fitness
        total_know = kb_stats["total_knowledge"]
        total_imp = kb_stats.get("total_importance", 0)

        # 构建 Markdown
        md = f"""# 蜂群吸收汇报

**时间**: {now.strftime('%Y-%m-%d %H:%M')}  
**世代**: {gen}  
**种群**: {alive_count}  

---

## 种群概况

| 指标 | 数值 |
|------|------|
| 存活原子 | {alive_count} |
| 活跃原子 | {active_count} |
| 平均适应度 | {avg_fitness:.3f} |
| 最高适应度 | {max_fitness:.3f} |

---

## 蜂种分布

"""

        # 按数量排序
        sorted_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)
        for role, count in sorted_species:
            emoji = self._get_role_emoji(role)
            md += f"- {emoji} {role}: {count}\n"

        md += f"""

---

## 知识库

| 指标 | 数值 |
|------|------|
| 总知识条目 | {total_know} |
| 知识重要性 | {total_imp:.2f} |

---

## 世代统计

| 指标 | 数值 |
|------|------|
| 新生 | {self.swarm.metrics.newborns} |
| 死亡 | {self.swarm.metrics.deaths} |
| 簇数量 | {self.swarm.metrics.clusters} |

---

*Generated at {now.isoformat()}*
"""

        return md

    def _get_role_emoji(self, role: str) -> str:
        """获取蜂种 emoji"""
        emoji_map = {
            "miner": "⛏️",
            "analyzer": "📊",
            "architect": "🏗️",
            "builder": "🔨",
            "orchestrator": "🎼",
            "deployer": "🚀",
            "engineer": "⚙️",
            "worker": "🐝",
            "sentinel": "🛡️",
            "librarian": "📚",
        }
        return emoji_map.get(role, "🐝")

    def save_report(self, filename: str = None) -> Path:
        """保存汇报到文件"""

        if filename is None:
            now = datetime.datetime.now()
            filename = f"report_{now.strftime('%Y%m%d_%H%M')}.md"

        filepath = self.output_dir / filename

        content = self.generate_report()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    def save_daily_index(self) -> Path:
        """保存每日索引"""

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        index_file = self.output_dir / f"index_{date_str}.md"

        # 查找当天的所有报告
        reports = list(self.output_dir.glob(f"report_{date_str}*.md"))
        reports.sort()

        md = f"""# 📅 {date_str} 蜂群汇报

## 当日汇报

"""

        for r in reports:
            time = r.stem.replace(f"report_{date_str}_", "")
            md += f"- [[{r.name}]] - {time}\n"

        md += f"""

---

*共 {len(reports)} 份汇报*
"""

        with open(index_file, "w", encoding="utf-8") as f:
            f.write(md)

        return index_file


class HourlyReporter:
    """每小时汇报定时器"""

    def __init__(self, swarm: Swarm, report_dir: str = None):
        self.reporter = SwarmReporter(swarm, report_dir)
        self.last_report_time = None
        self.report_interval = 3600  # 1小时

    def should_report(self) -> bool:
        """是否应该生成汇报"""
        import time

        if self.last_report_time is None:
            return True

        return time.time() - self.last_report_time >= self.report_interval

    def report_if_needed(self):
        """如果需要则生成汇报"""
        import time

        if self.should_report():
            filepath = self.reporter.save_report()
            self.last_report_time = time.time()
            print(f"📊 汇报已保存: {filepath}")
            return filepath

        return None


def start_hourly_reporting(swarm: Swarm, report_dir: str = None):
    """启动每小时汇报"""

    reporter = HourlyReporter(swarm, report_dir)

    # 生成第一份汇报
    reporter.report_if_needed()

    return reporter


if __name__ == "__main__":
    # 测试
    from hive.core.swarm import Swarm

    swarm = Swarm()

    # 进化几代
    for _ in range(3):
        swarm.evolve_once()

    # 生成汇报
    reporter = SwarmReporter(swarm)
    reporter.save_report()
    print(f"汇报已保存到: {reporter.output_dir}")
