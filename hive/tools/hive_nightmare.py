"""
HiveNightmare Inspector (VSS 权限提升与快照提权法医工具)
设计灵感来源于 CVE-2021-36934 (HiveNightmare)。
这套探针能够绕开蜂后(QueenBee)的视线和底层 try-catch 被截断的迷雾。
如果出现工蜂神秘失联、物理性暴死、OOM 坠落，主控可以使用此探针强行提纯该工蜂坠亡前的日志文件（death_certificates）
抽提出最致命的崩溃信息，从而为顶级安全管理员 (Architect/Warden) 提供最高权限的透明视界。
"""

import os
import json
import glob
import logging
from logging import getLogger
from pathlib import Path

logger = getLogger("HiveNightmare")
logging.basicConfig(level=logging.INFO, format='%(message)s')

class HiveNightmareInspector:
    def __init__(self, workspace_path: str = "d:/Hive/Data"):
        # 利用最高权限 (这里假设执行脚本的有这个路径的 r 权限) 渗透入系统沙盒内
        self.workspace = Path(workspace_path)
        self.cert_dir = self.workspace / "hive" / "morgue"
        
        logger.info("[+] HiveNightmare 探针已注入。拥有越级目录遍历权限。")
        if not self.cert_dir.exists():
            logger.warning("[-] 未发现墓碑(VSS快照)根目录。")

    def dump_shadow_copy(self, worker_id: str = None) -> list[str]:
        """
        强制读取隐藏目录下的 VSS/墓碑 快照。
        如果不提供 worker_id，则抓取最近的一份。
        """
        logger.info(f"[*] 正在爆破 VSS 存储域: {self.cert_dir}")
        if not self.cert_dir.exists():
            return []
            
        search_pattern = f"death_cert_*_{worker_id}_*.json" if worker_id else "death_cert_*.json"
        snapshots = list(self.cert_dir.glob(search_pattern))
        
        # 按时间排序取最近的快照
        snapshots.sort(key=os.path.getmtime, reverse=True)
        return snapshots

    def analyze_last_breath(self, snapshot_path: Path):
        """
        将一具冰冷的尸骸反编撰为生前影像。提权拉取内部堆栈详情。
        """
        logger.info(f"[*] 发现残存快照。开始提取 SAM 哈希与生命迹象...")
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            print("\n" + "="*50)
            print(" 🩸 [HIVE NIGHTMARE] VSS FORENSIC REPORT 🩸 ")
            print("="*50)
            print(f"💀 受害者: {data.get('worker_id', 'UNKNOWN')}")
            print(f"⏱️ 暴亡时间: {data.get('timestamp')}")
            print(f"🗡️ 致命凶器 (Error): {data.get('error')}")
            
            dna = data.get("dna", {})
            print(f"\n🧬 工蜂死前持有 DNA (装备):")
            print(f"  - 物种: {dna.get('species')}")
            print(f"  - 目标: {dna.get('target_name')}")
            print(f"  - 挂载基因: {dna.get('mutation_gene')}")
            
            tb = data.get("traceback", "")
            if tb:
                print("\n💥 系统内存崩溃栈帧快照:")
                print("------------- TRACEBACK START -------------")
                # 为防止刷屏只打印前最后 5 行，或者全部
                lines = tb.strip().split("\n")
                if len(lines) > 10:
                    print("... (中间追踪断层) ...")
                    print("\n".join(lines[-10:]))
                else:
                    print(tb)
                print("-------------- TRACEBACK END --------------")
                
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"[-] 提权读取快照失败，数据可能损毁: {e}")
