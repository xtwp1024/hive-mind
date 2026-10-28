#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
演示 HiveNightmare (上帝视角的法医诊断室) 运作效果的沙盘推演脚本。
"""

import time
import os
import shutil
import threading
from hive.core.queen import QueenBee
from hive.core.dna import BeeDNA
from hive.tools.hive_nightmare import HiveNightmareInspector

def suicide_mission():
    """抛弃多余保护膜，我们将直接借用死亡开具大夫生成逼真墓碑。"""
    print("\n[Simulation] 投放致命任务：命令工蜂飞向不存在的零维度空间并除以零...")
    
    from hive.bees.base_bee import BaseBee
    from hive.core.diagnostics import create_death_certificate
    from hive.config.settings import get_settings
    import multiprocessing
    import time
    
    print(f"[SuicideBee] 正在接近坐标 BlackHole_Sector_Z...")
    time.sleep(1)
    
    try:
        print(f"[SuicideBee] 遭遇时空风暴，正在执行 1 / 0 的跨维计算...")
        _ = 1 / 0
    except Exception as e:
        q = multiprocessing.Queue()
        dna = BeeDNA(species="Standard", target_name="BlackHole_Sector_Z", mutation_gene="Deep_Space_Probe")
        bee = BaseBee(dna=dna, id="Victim_001", result_queue=q)
        
        # 将尸骨沉入停尸房
        cert = create_death_certificate(bee, e)
        cert_path = cert.save(get_settings().BASE_DIR / "hive" / "morgue")
        print(f"💀 [SYSTEM] 一具被时空风暴撕裂的工蜂残存记忆碎片已经无情地散落在了 {cert_path}")

def run_investigation():
    # 清理掉之前可能留下的死因碎片历史
    if os.path.exists("workspace-dev/diagnostics/death_certificates"):
        shutil.rmtree("workspace-dev/diagnostics/death_certificates")
        
    # 我们在新线程发起这个让进程坠毁的任务
    # 强制在主线程直接跑，这样它的崩溃就能直接反映在当场
    import traceback
    try:
        suicide_mission()
    except Exception as e:
        with open("nightmare_dump.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"[!] 脚本层直接崩解并未触达底层异常接管: {e}")
    
    print("\n[!] 警报！工蜂 Victim_001 信号丢失。没有返回任何生命迹象！常规队列未能捕获异常！")
    print("\n[+] 全局网管介入，拉起 HiveNightmareInspector 进行降维强力探查...")
    
    # 使用新长出的獠牙深挖底线信息
    inspector = HiveNightmareInspector()
    snapshots = inspector.dump_shadow_copy() # 搜索全部死亡快照捞取最近一个
    
    if not snapshots:
        print("[-] VSS 空洞。并未发现最近死亡痕迹。")
    else:
        # 直接拿最近的
        inspector.analyze_last_breath(snapshots[0])

if __name__ == "__main__":
    run_investigation()
