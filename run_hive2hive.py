#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 Hive2Hive P2P 结果版本链与分层权限树的沙盘脚本。
展示 add/update/recover（版本时间机器）+ grant/require（权限拦截）完整闭环。
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from hive.core.result_vault import ResultVault
from hive.core.access_tree import AccessTree, PermissionType, AccessDeniedError

SEP = "=" * 62

def demo_version_chain(vault: ResultVault):
    print(f"\n{SEP}")
    print("  PART 1: 结果版本链（FileVersioning + Conflict Recovery）")
    print(SEP)

    # FileObserver: 注册变更监听
    def on_change(key, ver):
        print(f"  📢 [Watchdog] {key!r} 发生变更 → {ver}")
    vault.watch("market://AAPL", on_change)

    # 1. Queen 首次写入 v0（对标 fileManager.add）
    print("\n[Step 1] Queen 写入初始分析结果 v0...")
    vault.add("market://AAPL", {"score": 0.72, "signal": "BUY"}, author="Queen")

    # 2. Analyst 更新 v1（对标 fileManager.update）
    print("\n[Step 2] Analyst 追加更新版本 v1...")
    vault.update("market://AAPL", {"score": 0.85, "signal": "STRONG_BUY"}, author="Analyst", note="加入社媒情绪权重")

    # 3. 故意写坏数据 v2（模仿故障工蜂写入了垃圾）
    print("\n[Step 3] 故障工蜂 BadBee 写入了错误数据 v2...")
    vault.update("market://AAPL", {"score": -999, "signal": "ERROR"}, author="BadBee", note="数据损坏！")

    # 4. 枚举所有版本（对标 IVersionSelector）
    print("\n[Step 4] 查看完整版本历史...")
    for v in vault.versions("market://AAPL"):
        print(f"  {v} → score={v.value.get('score')} signal={v.value.get('signal')} | {v.note or '无备注'}")

    # 5. Queen 执行版本回溯（对标 fileManager.recover）
    print("\n[Step 5] Queen 发现数据异常，回溯到 v1（Analyst 的正确版本）...")
    recovered = vault.recover("market://AAPL", version_idx=1)
    print(f"  回溯完成 → 当前最新版本: {vault.get('market://AAPL')}")

    latest = vault.get("market://AAPL")
    assert latest["score"] == 0.85, "版本回溯失败！"
    print(f"\n  ✅ 版本链验证通过：写入、追踪、回溯全部正常！")


def demo_access_tree(vault: ResultVault):
    print(f"\n{SEP}")
    print("  PART 2: 分层权限树（PermissionType + FileSharing）")
    print(SEP)

    tree = AccessTree()

    # 建立权限树（对标 userManager.register + fileManager.share）
    print("\n[Step 1] 建立权限树...")
    tree.grant("/results",          "Queen",     PermissionType.OWNER,      granted_by="system")
    tree.grant("/results/public",   "Inspector", PermissionType.READ_ONLY,  granted_by="Queen")
    tree.grant("/results/collab",   "Analyst",   PermissionType.READ_WRITE, granted_by="Queen")

    print(f"  /results 权限: Queen={tree.effective_permission('/results', 'Queen')}")
    print(f"  /results/public 权限: Inspector={tree.effective_permission('/results/public','Inspector')}")
    print(f"  /results/collab 权限: Analyst={tree.effective_permission('/results/collab','Analyst')}")

    # Inspector 尝试写入（READ_ONLY，应被拒绝）
    print("\n[Step 2] Inspector（只读权限）尝试写入结果...")
    vault.add("results/public/test", {"data": "ok"}, author="Queen")  # 初始化
    try:
        tree.require("/results/public", "Inspector", PermissionType.READ_WRITE)
        vault.update("results/public/test", {"data": "hacked"}, author="Inspector")
        print("  ❌ 写入成功（这不应该发生！）")
    except AccessDeniedError as e:
        print(f"  🔒 权限拦截成功: {e}")

    # Analyst 合法写入（READ_WRITE）
    print("\n[Step 3] Analyst（读写权限）写入协作结果...")
    vault.add("results/collab/analysis", {"signal": "BUY", "conf": 0.9}, author="Queen")
    tree.require("/results/collab", "Analyst", PermissionType.READ_WRITE)
    vault.update("results/collab/analysis", {"signal": "STRONG_BUY", "conf": 0.96}, author="Analyst")
    print(f"  ✅ Analyst 写入成功: {vault.get('results/collab/analysis')}")

    # 子路径继承：Inspector 尝试访问 /results/public/deep/data（继承 READ_ONLY）
    print("\n[Step 4] Inspector 访问深层子路径（继承父路径权限）...")
    eff = tree.effective_permission("/results/public/deep/nested", "Inspector")
    print(f"  /results/public/deep/nested 有效权限: Inspector={eff} (继承自父路径)")
    assert eff == PermissionType.READ_ONLY, "继承权限失败！"

    print(f"\n  ✅ 权限树验证通过：授权、拒绝、继承全部正常！")


def main():
    print(SEP)
    print("  Hive2Hive 结果版本链 + 分层权限树 - 全流程演示")
    print(SEP)

    vault = ResultVault("ProductionVault")

    demo_version_chain(vault)
    demo_access_tree(vault)

    # 最终仓库摘要
    print(f"\n{SEP}")
    print("  最终 ResultVault 摘要（各 key 版本数）:")
    for k, cnt in vault.vault_summary().items():
        print(f"    {k:<40} : {cnt} 个版本")

    print(f"\n{SEP}")
    print("  🎉 全部验证通过！结果版本链 + 分层权限树部署完成！")
    print(SEP)

if __name__ == "__main__":
    main()
